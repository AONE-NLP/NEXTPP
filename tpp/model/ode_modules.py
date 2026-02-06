import torch
import torch.nn as nn
import torchdiffeq

class NeuralODE(nn.Module):
    """
    Neural ODE module that defines the continuous dynamics of the system.
    Implements dz/dt = fθ(z,t) where fθ is a neural network.
    Uses tanh activation and layer normalization for stable training.
    """
    def __init__(self, d_z, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_z)
        )
    
    def forward(self, t, z):
        return self.net(z)

class LatentSpace(nn.Module):
    """
    Latent space encoder that processes input embedding using Transformer layers.
    Generates stochastic latent representations (γ, logvar) for each event.
    """
    def __init__(self, d_z, d_model, n_attn_heads=1, n_tf_layers=1, dropout_prob=0.0):
        super().__init__()
        self.transformer_stack = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_attn_heads,
                dim_feedforward=2 * d_model,
                batch_first=True,
                dropout=dropout_prob,
            ) for _ in range(n_tf_layers)
        ])
        self.gamma_proj = nn.Linear(d_model, d_z)
        self.tau_proj = nn.Linear(d_model, d_z)
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        batch_size = x.shape[0]
        agg_tokens = self.agg_token.expand(batch_size, -1, -1)
        x = torch.cat([x, agg_tokens], dim=1)
        x = self.transformer_stack(x)
        gamma = self.gamma_proj(x[:, -1, :])
        logvar = self.tau_proj(x[:, -1, :])
        return gamma, logvar
  
class ODEEncoder(nn.Module):
    """
    Main ODE Encoder module processing:
    1. Divides input sequence into temporal chunks(This step is unnecessary when chunking is not applied.)
    2. Encodes each envent(or chunk) into latent variables (γ, τ)
    3. Evolves latent states through neural ODE between envents(or chunks)
    4. Computes consistency losses (MSE between envents(or chunks) and KL regularization)
    """
    def __init__(self, d_model, init_chunk_ratio=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_z = self.d_model // 4
        self.ode_func = NeuralODE(self.d_z, d_model)
        self.projection = nn.Linear(self.d_z, self.d_model)
        self.latent_repsent = LatentSpace(self.d_z, d_model)
        self.chunk_ratio = init_chunk_ratio
    
    def get_chunk_size(self, seq_length):
        """Computes chunk size based on ratio or fixed size"""
        if self.chunk_ratio > 0:
            return max(1, min(int(seq_length * self.chunk_ratio), seq_length))
        return self.chunk_size

    def forward(self, latent_rep, time_seqs):
        """
        Processes input sequence through ODE evolution:
        1. Normalizes timestamps to [0,1] range per sequence
        2. For each envent(or chunk):
           - Samples initial state z0 ~ N(γ, τ²)
           - Solves ODE from envent(or chunk) start to next envent(or chunk)'s start time
           - Projects evolved state back to original dimension
        3. Computes losses:
           - MSE between predicted and actual next envent(or chunk)
           - KL divergence from prior N(0,I)
        """
        B, L, D = latent_rep.shape
        chunk_size = self.get_chunk_size(L)
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, L)
            chunks.append(latent_rep[:, start:end, :])
        
        gamma_list, logvar_list, std_list = [], [], []
        for chunk in chunks:
            gamma, logvar = self.latent_repsent(chunk)
            gamma_list.append(gamma)
            logvar_list.append(logvar)
            std = torch.exp(0.5 * logvar)
            std_list.append(std)
        
        gamma_all = torch.stack(gamma_list, dim=1)
        logvar_all = torch.stack(logvar_list, dim=1)
        tau_all = torch.stack(std_list, dim=1) 
        
        # Normalize timestamps to [0,1] range
        min_vals, _ = time_seqs.min(dim=1, keepdim=True)
        max_vals, _ = time_seqs.max(dim=1, keepdim=True)
        denominator = torch.where(
            (max_vals - min_vals) < 1e-6,
            torch.ones_like(max_vals),
            max_vals - min_vals
        )
        norm_time = (time_seqs - min_vals) / denominator
        
        outputs = []
        kl_losses = []
        next_chunk_start_times = []
        
        # Compute boundary times between envent(or chunk)
        for i in range(num_chunks):
            next_idx = min((i + 1) * chunk_size, L - 1)
            next_chunk_start_times.append(norm_time[:, next_idx])
        next_chunk_start_times = torch.stack(next_chunk_start_times, dim=1)
        
        mse_losses = []
        for i in range(num_chunks):
            # ODE evolution between envent(or chunk) boundaries
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, L)
            actual_size = end_idx - start_idx
            
            if actual_size == 0:
                continue
                
            t_start = norm_time[:, start_idx]
            # t_end = next_chunk_start_times[:, i] if i < num_chunks - 1 else torch.diff(norm_time[:, -3:], dim=1).mean(dim=1)
            t_end = next_chunk_start_times[:, i] if i < num_chunks - 1 else norm_time[:, end_idx - 1]
            durations = (t_end - t_start).clamp(min=1e-6)
            
            # Sample initial state and evolve through ODE
            z_current = gamma_all[:, i] + tau_all[:, i] * torch.randn_like(tau_all[:, i])
            
            def scaled_ode(t, z):
                """Time-scaled ODE function that accounts for chunk duration"""
                return self.ode_func(t, z) * durations.unsqueeze(-1)
            
            '''
            The ODE latent space evolution from T[strat] to T[end] maps time to the interval [0,1],
            with the ODE's timescale correspondingly scaled to the distance between T[strat] and T[end].
            '''
            t_range = torch.tensor([0.0, 1.0], device=latent_rep.device)
            z_out = torchdiffeq.odeint(scaled_ode, z_current, t_range, method='rk4')[-1]
            
            # Decoder: Project evolved state back to original dimension
            context = self.projection(z_out).unsqueeze(1).expand(-1, actual_size, -1)
            outputs.append(context)
            
            # Compute MSE loss 
            if i < num_chunks-1:
                if i == num_chunks-2 and L%chunk_size != 0:
                    context = context[:,:L%chunk_size,:]
                mse_per_sample = torch.mean(torch.square(chunks[i+1] - context))
                mse_losses.append(mse_per_sample)
            
            # Compute KL divergence 
            pz0_mean = pz0_logvar = torch.zeros(z_out.size()).to(latent_rep.device)
            if i < num_chunks:
                kl_loss = self.normal_kl(
                    gamma_all[:, i], 
                    logvar_all[:, i],
                    pz0_mean,
                    pz0_logvar
                ).mean()
                kl_losses.append(kl_loss)
        
        output_tensor = torch.cat(outputs, dim=1)
        raw_mse_loss = torch.stack(mse_losses).mean() if mse_losses else 0.0
        raw_kl_loss = torch.stack(kl_losses).mean() if kl_losses else 0.0
        total_loss_value = raw_mse_loss + raw_kl_loss

        return output_tensor, total_loss_value

    def normal_kl(self, mu1, lv1, mu2, lv2):
        """Computes KL divergence between two normal distributions"""
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        return kl