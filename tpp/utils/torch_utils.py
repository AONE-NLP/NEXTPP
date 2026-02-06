import os
import random
import numpy as np
import torch
from tpp.utils.import_utils import is_torch_mps_available
from omegaconf import OmegaConf  

def set_seed(seed=1029):
    """Set global random seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if is_torch_mps_available():
        torch.mps.manual_seed(seed)

def set_device(gpu=-1):
    """Configure computing device"""
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    elif gpu >= 0 and is_torch_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_optimizer(optimizer, params, lr, **kwargs):
    """Set up optimizer"""
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, **kwargs)
        elif optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, **kwargs)
        elif optimizer == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, **kwargs)
        elif optimizer == "adagrad":
            return torch.optim.Adagrad(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    elif callable(optimizer):
        return optimizer(params, lr=lr, **kwargs)
    
    elif isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    
    else:
        raise TypeError("Invalid optimizer type. Must be string, class or Optimizer instance")

def count_model_params(model, verbose=False):
    """Count model parameters"""
    total_params = 0
    trainable_params = 0
    param_details = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        is_trainable = param.requires_grad
        
        if is_trainable:
            trainable_params += num_params
        
        if verbose:
            param_details.append({
                "name": name,
                "shape": tuple(param.shape),
                "num_params": num_params,
                "trainable": is_trainable
            })
    
    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
    }
    
    if verbose:
        stats["param_details"] = param_details
    
    return stats

# Utility functions
def conf_to_dict(conf):
    return OmegaConf.to_container(conf, resolve=True) if OmegaConf.is_config(conf) else conf

def safe_serialize(data):
    return conf_to_dict(data)