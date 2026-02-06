import torch
import torch.nn as nn

from tpp.model.baselayer import MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus ,EncoderLayer,attention
from tpp.model.basemodel import TorchBaseModel
from tpp.model.ode_modules import ODEEncoder


class NEXTPP(TorchBaseModel):
    """
    Neural Evolution for X-Interaction framework in Marked Temporal Point Processes.(NEXTPP)
    Combines neural ODEs with attention mechanisms for temporal event prediction.
    """
    def __init__(self, model_config):
        """
        Initialize NEXTPP model components.
        
        Args:
            model_config: Configuration object containing:
                - hidden_size: Dimension of hidden representations
                - time_emb_size: Dimension of temporal embeddings
                - use_ln: Whether to use layer normalization
                - chunk_ratio: Ratio for sequence chunking (0-1. When the size is sufficiently small, chunking is unnecessary.)
                - num_layers: Number of transformer layers
                - num_heads: Number of attention heads
                - dropout_rate: Dropout probability
        """
        super(NEXTPP, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln
        self.chunk_ratio = model_config.chunk_ratio
        self.d_rnn = model_config.hidden_size // 2
        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        # Temporal encoding layer (converts timestamps to vectors)
        self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

        # Intensity function parameters
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        
        # Xavier initialization for intensity parameters
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # Intensity computation layers
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)   

        # Feed-forward network for attention blocks
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # Core model components
        self.ode_encoder = ODEEncoder(
            d_model=self.d_model, 
            init_chunk_ratio=self.chunk_ratio
        )
        self.encoder = EncoderLayer(
            self.d_model,
            MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                             output_linear=False),
            use_residual=False,
            feed_forward=self.feed_forward,
            dropout=self.dropout
        ) 
        
    def forward(self, time_seqs, type_seqs, attention_mask):
        """
        Forward pass through NEXTPP model.
        
        Args:
            time_seqs: Batch of event timestamps [batch_size, seq_len]
            type_seqs: Batch of event types [batch_size, seq_len]
            attention_mask: Boolean mask for valid events
            
        Returns:
            output: Final hidden representations [batch_size, seq_len, d_model]
        """
        # Temporal and type embeddings
        tem_enc = self.layer_temporal_encoding(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)
        
        # Combine embeddings and process through encoder
        enc_output += tem_enc
        env_emb = self.encoder(enc_output, attention_mask)

        # ODE-based temporal evolution 
        enc_output, ode_loss = self.ode_encoder(enc_output, time_seqs)
        self.ode_loss = ode_loss  

        # CA
        output, attn = attention(enc_output, env_emb, env_emb, mask=attention_mask)
        
        return output

    
    
    
    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])

        # [batch_size, seq_len, num_event_types]
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        type_seq=type_seqs[:, 1:])

        # compute loss to minimize
        # print("- (event_ll - non_event_ll).sum():   ",- (event_ll - non_event_ll).sum(),"ode_loss:  ",self.ode_loss)
        loss = - (event_ll - non_event_ll).sum() #+ self.ode_loss
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
