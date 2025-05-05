from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    input_dim: int  # set by Config
    latent_dim: int = 256

    codebook_entries: List[int] = [20, 40]

    # Encoder/Decoder
    dropout: float = 0.0
    batch_norm: bool = False
    enc_hidden_dims: List[int] = [1024, 512]
    dec_hidden_dims: List[int] = [512, 1024]
    activation: str = "relu"

    # Loss
    commitment_beta: float = 0.25
