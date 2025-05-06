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

        self.codes = nn.Parameter(torch.rand(codebook_size, latent_dim))

    def forward(self, z: torch.Tensor):
        """
        z: (B, D) — input vectors to be quantized
        returns:
            z_q: quantized output (B, D)
            loss: VQ + commitment loss
            indices: selected code indices (B,)
        """
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

        if self.training:
            entropy_weight = 0.1
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
            avg_probs = one_hot.mean(dim=0)
            entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
            loss = loss - entropy_weight * entropy

        z_out = z + (z_q - z).detach()

        out = VQOutput(
            z_q=z_out,
            loss=loss,
            indices=indices,
        )
        return out
