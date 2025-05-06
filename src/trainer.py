import os
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from configs.config import Config
from src.datasets.course_dataset import CourseDataset, course_collate_fn
from src.models.rqvae import RQVAE


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        full_dataset = CourseDataset(cfg=cfg.data)
        total_size = len(full_dataset)
        val_size = int(cfg.train.val_split * total_size)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # reproducibility
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=course_collate_fn,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            collate_fn=course_collate_fn,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
        )

        # Model
        self.rqvae = RQVAE(cfg.model).to(self.device)
        self.optim = torch.optim.Adam(self.rqvae.parameters(), lr=cfg.train.init_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=self.cfg.train.epochs * len(self.train_loader),
            eta_min=cfg.train.min_lr,
        )

        self.ep = 0
        self.train_losses = []
        self.val_losses = []

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
            codebook_usages = [Counter() for _ in range(len(self.rqvae.vqs))]
            for data in tqdm(
                self.train_loader, desc=f"Epoch {self.ep} / {self.cfg.train.epochs}"
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

                for i in range(out.indices.shape[1]):
                    codebook_usages[i].update(out.indices[:, i].cpu().tolist())

            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            if (self.ep % self.cfg.train.eval_every == 0) or (
                self.ep == self.cfg.train.epochs
            ):
                val_loss = self.eval()
                self.val_losses.append(val_loss)
                print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {avg_train_loss:.4f}")
            if (self.ep % self.cfg.train.save_every == 0) or (
                self.ep == self.cfg.train.epochs
            ):
                self.save_model()

            reinit_indices = []
            for i, counter in enumerate(codebook_usages):
                unused = [
                    k for k in range(self.rqvae.vqs[i].codebook_size) if counter[k] == 0
                ]
                reinit_indices.append(
                    torch.tensor(unused, device=self.device, dtype=torch.long)
                )

            self.rqvae.reinit_codes(reinit_indices)

        torch.save(
            {"train_losses": self.train_losses, "val_losses": self.val_losses},
            os.path.join(self.outputs_dir, "metrics.pth"),
        )
        print("Training completed.")

    def eval(self):
        self.rqvae.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                x = data["embedding"].to(self.device)
                out = self.rqvae(x)
                loss = out.loss.mean()
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_model(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{self.ep}.pth")
        torch.save(self.rqvae.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
