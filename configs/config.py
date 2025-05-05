from dataclasses import dataclass 
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.configs.data_config import DataConfig

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
