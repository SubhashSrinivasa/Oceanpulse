"""ConvLSTM cell (Shi et al. 2015), peephole-free."""
from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        # GroupNorm with 4 groups (one per gate: i/f/g/o). Normalises within each
        # gate's channels independently of spatial size, so tiles and full grid both work.
        self.norm = nn.GroupNorm(4, 4 * hidden_channels, affine=True)

    def init_state(self, batch: int, h: int, w: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch, self.hidden_channels, h, w, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        gates = self.norm(self.conv(torch.cat([x, h_prev], dim=1)))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)
