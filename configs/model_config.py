from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    input_dim: int  # set by Config
    latent_dim: int = 128

    codebook_sizes: List[int] = field(default_factory=lambda: [10, 5, 5])

    # Encoder/Decoder
    dropout: float = 0.1
    batch_norm: bool = True
    enc_hidden_dims: List[int] = field(default_factory=lambda: [512])
    dec_hidden_dims: List[int] = field(default_factory=lambda: [512])
    activation: str = "relu"

    # Loss
    commitment_beta: float = 0.15
