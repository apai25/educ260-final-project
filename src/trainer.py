import os
from datetime import datetime

import torch
from tqdm import tqdm

from configs.config import Config
from src.datasets.course_dataset import CourseDataset, course_collate_fn
from src.models.rqvae import RQVAE


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data
        self.dataset = CourseDataset(
            cfg=cfg.data,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=course_collate_fn,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
        )

        # Model
        self.rqvae = RQVAE(cfg.model).to(self.device)
        self.optim = torch.optim.Adam(self.rqvae.parameters(), lr=cfg.train.init_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=self.cfg.train.epochs * len(self.dataloader),
            eta_min=cfg.train.min_lr,
        )
        self.loss_fn = torch.nn.MSELoss()

        # Init training vars, dirs
        self.ep = 0
        self.train_losses = []

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.outputs_dir = os.path.join(cfg.train.outputs_dir, timestamp)
        self.checkpoint_dir = os.path.join(self.outputs_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save configs
        torch.save(
            vars(cfg.train), os.path.join(self.outputs_dir, "trainer_config.pth")
        )
        torch.save(vars(cfg.model), os.path.join(self.outputs_dir, "model_config.pth"))
        torch.save(vars(cfg.data), os.path.join(self.outputs_dir, "data_config.pth"))

        print(f"Device: {self.device}")
        print(f"Outputs directory: {self.outputs_dir}")

    def train(self):
        print("Starting training...")
        for ep in range(self.cfg.train.epochs):
            self.rqvae.train()
            self.ep = ep + 1

            train_loss = 0.0
            for data in tqdm(
                self.dataloader, desc=f"Epoch {self.ep} / {self.cfg.train.epochs}"
            ):
                x = data["embedding"].to(self.device)
                out = self.rqvae(x)

                loss = out.loss.mean()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.rqvae.parameters(), self.cfg.train.grad_clip
                )
                self.optim.step()
                self.scheduler.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(self.dataloader)
            self.train_losses.append(avg_loss)

            print(f"Epoch {self.ep} - Train Loss: {avg_loss:.4f}")
            self.save_model()

        torch.save(
            {"train_losses": self.train_losses},
            os.path.join(self.outputs_dir, "metrics.pth"),
        )
        print("Training completed.")

    def save_model(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{self.ep}.pth")
        torch.save(self.rqvae.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
