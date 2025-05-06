from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 30
    init_lr: float = 1e-3
    min_lr: float = 1e-5
    grad_clip: float = 1.0

    val_split: float = 0.2

    num_workers: int = 64
    pin_memory: bool = True

    outputs_dir: Path = Path("outputs").resolve()
    save_every: int = 10
    eval_every: int = 5
