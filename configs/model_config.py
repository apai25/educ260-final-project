from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    input_dim: int  # set by Config
    latent_dim: int = 256

    codebook_sizes: List[int] = field(default_factory=lambda: [50, 300])

    # Encoder/Decoder
    dropout: float = 0.0
    batch_norm: bool = False
    enc_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    dec_hidden_dims: List[int] = field(default_factory=lambda: [512, 1024])
    activation: str = "relu"

    # Loss
    commitment_beta: float = 0.25
