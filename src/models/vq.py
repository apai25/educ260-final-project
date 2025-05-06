from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor


class VQ(nn.Module):
    def __init__(
        self, codebook_size: int, latent_dim: int, commitment_beta: float = 0.25
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta

        self.codes = nn.Parameter(torch.randn(codebook_size, latent_dim))

    def forward(self, z: torch.Tensor):
        dists = (
            z.pow(2).sum(dim=1, keepdim=True)
            - 2 * z @ self.codes.t()
            + self.codes.pow(2).sum(dim=1)
        )

        indices = torch.argmin(dists, dim=1)
        z_q = self.codes[indices]

        vq_loss = F.mse_loss(z_q, z.detach(), reduction="none")
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction="none")
        loss = torch.mean(vq_loss + self.commitment_beta * commitment_loss, dim=1)

        z_out = z + (z_q - z).detach()

        out = VQOutput(
            z_q=z_out,
            loss=loss,
            indices=indices,
        )
        return out

    def reinit_codes(self, indices: torch.Tensor):
        with torch.no_grad():
            self.codes.data[indices] = (
                torch.randn_like(self.codes.data[indices]) * 2
            )
