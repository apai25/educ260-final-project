from dataclasses import dataclass, field

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig


@dataclass
class Config:
    model: ModelConfig = field(init=False)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        self.model = ModelConfig(input_dim=self.data.embed_dim)
