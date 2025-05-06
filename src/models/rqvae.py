from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from configs.model_config import ModelConfig
from src.models.mlp import MLP
from src.models.vq import VQ


@dataclass
class RQVAEOutput:
    x_hat: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor


class RQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(RQVAE, self).__init__()
        self.cfg = cfg

        self.enc = MLP(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.enc_hidden_dims,
            output_dims=cfg.latent_dim,
            activation=cfg.activation,
            batch_norm=cfg.batch_norm,
            dropout=cfg.dropout,
        )

        self.vqs = nn.ModuleList(
            [
                VQ(
                    codebook_size=codebook_size,
                    latent_dim=cfg.latent_dim,
                    commitment_beta=cfg.commitment_beta,
                )
                for codebook_size in cfg.codebook_sizes
            ]
        )

        self.dec = MLP(
            input_dim=cfg.latent_dim,
            hidden_dims=cfg.dec_hidden_dims,
            output_dims=cfg.input_dim,
            activation=cfg.activation,
            batch_norm=cfg.batch_norm,
            dropout=cfg.dropout,
        )

        self.layer_norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, x):
        z = self.enc(x)
        z = self.layer_norm(z)

        z_q_total = torch.zeros_like(z)
        vq_outs = []
        residual = z
        for vq in self.vqs:
            vq_out = vq(residual)
            z_q = vq_out.z_q
            residual = residual - z_q.detach()
            z_q_total = z_q_total + z_q
            vq_outs.append(vq_out)

        x_hat = self.dec(z_q_total)

        recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
        vq_losses = torch.stack([vq_out.loss for vq_out in vq_outs], dim=1)
        loss = torch.sum(vq_losses, dim=1) + recon_loss

        indices = torch.stack([vq_out.indices for vq_out in vq_outs], dim=1)

        out = RQVAEOutput(
            x_hat=x_hat,
            loss=loss,
            indices=indices,
        )
        return out

    def reinit_codes(self, indices: List[torch.Tensor]):
        assert len(indices) == len(self.vqs), "Mismatch in codebooks vs indices"
        for vq, idx in zip(self.vqs, indices):
            if len(idx) > 0:
                vq.reinit_codes(idx)
