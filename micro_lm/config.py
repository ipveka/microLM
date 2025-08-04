import torch
from dataclasses import dataclass, asdict, field

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True

@dataclass
class TrainConfig:
    """Configuration for training."""
    # Data
    dataset: str = 'openwebtext'
    batch_size: int = 64
    
    # Model
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # Training
    learning_rate: float = 1e-3
    max_iters: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Logging
    eval_interval: int = 250
    log_interval: int = 10
    eval_iters: int = 200
    wandb_log: bool = False
    wandb_project: str = 'microLM'
    wandb_run_name: str = 'run'
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile: bool = False
    
    # Checkpointing
    out_dir: str = 'out'

# --- Pre-defined Configurations ---

def get_tinystories_config():
    """Returns a configuration for training on the TinyStories dataset."""
    model_cfg = ModelConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128,
        vocab_size=10000, # Smaller vocab for a smaller dataset
        dropout=0.1
    )
    train_cfg = TrainConfig(
        dataset='TinyStories',
        model_config=model_cfg,
        batch_size=32,
        learning_rate=5e-4,
        max_iters=10000,
        eval_interval=500
    )
    return train_cfg

def get_base_config():
    """Returns a base configuration for a medium-sized model."""
    model_cfg = ModelConfig() # Uses default values
    train_cfg = TrainConfig(
        model_config=model_cfg
    )
    return train_cfg

def get_config(name: str = 'base'):
    """
    Returns a configuration by name.
    
    Args:
        name (str): The name of the configuration to return.
                    Options: 'base', 'tinystories'.
    
    Returns:
        TrainConfig: The requested training configuration.
    """
    if name == 'tinystories':
        return get_tinystories_config()
    elif name == 'base':
        return get_base_config()
    else:
        raise ValueError(f"Unknown configuration name: {name}")

if __name__ == '__main__':
    # Example of how to use the configurations
    base_config = get_config('base')
    print("Base Config:", asdict(base_config))
    
    tinystories_config = get_config('tinystories')
    print("\nTinyStories Config:", asdict(tinystories_config))