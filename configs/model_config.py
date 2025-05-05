from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    input_dim: int # set by Config
    latent_dim: int = 256

    codebook_entries: List[int] = [20, 40]

    dropout: float = 0.0
    batch_norm: bool = False
