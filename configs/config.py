from dataclasses import dataclass

from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
