"""Encoder-decoder ConvLSTM for multi-step ocean forecasting.

Encoder reads INPUT_LEN frames of (C_in=10) inputs. Its final hidden/cell
states seed the decoder, which autoregressively emits OUTPUT_LEN frames of
C_tgt=5 target channels. Decoder input at each step is the previous target
prediction concatenated with the 3 auxiliary channels (sin_doy, cos_doy,
ocean_mask) for the current future date.

Residual prediction (residual=True, default): the head outputs a delta
(Δy = y_pred - init_tgt) rather than an absolute value. The delta is added
to init_tgt to get the absolute prediction, which is then fed back as the
next decoder input. At random init the head outputs ≈ 0, so the model starts
from persistence and learns deviations — a much better inductive bias than
learning absolute ocean values from scratch. The public interface (forward,
decode) still returns absolute predictions, so train/evaluate/forecast are
unchanged.

Scheduled sampling (train mode): with probability p the decoder is fed the
ground-truth target at step t-1 instead of its own prediction; p is ramped
during training.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.convlstm_cell import ConvLSTMCell


class ConvLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        *,
        n_input_channels: int,
        n_target_channels: int,
        n_aux_channels: int,
        hidden_channels: tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.n_input = n_input_channels
        self.n_target = n_target_channels
        self.n_aux = n_aux_channels
        self.hidden_channels = tuple(hidden_channels)
        self.n_layers = len(self.hidden_channels)
        self.residual = residual

        enc_ch = [n_input_channels, *self.hidden_channels[:-1]]
        self.encoder = nn.ModuleList(
            ConvLSTMCell(enc_ch[i], self.hidden_channels[i], kernel_size)
            for i in range(self.n_layers)
        )

        dec_in = n_target_channels + n_aux_channels
        dec_ch = [dec_in, *self.hidden_channels[:-1]]
        self.decoder = nn.ModuleList(
            ConvLSTMCell(dec_ch[i], self.hidden_channels[i], kernel_size)
            for i in range(self.n_layers)
        )

        self.head = nn.Conv2d(self.hidden_channels[-1], n_target_channels, kernel_size=1)

    def _init_states(self, batch: int, h: int, w: int, device, dtype, cells: nn.ModuleList):
        return [cell.init_state(batch, h, w, device, dtype) for cell in cells]

    def encode(self, x_seq: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # x_seq: (B, T, C, H, W)
        B, T, _, H, W = x_seq.shape
        states = self._init_states(B, H, W, x_seq.device, x_seq.dtype, self.encoder)
        for t in range(T):
            inp = x_seq[:, t]
            for li, cell in enumerate(self.encoder):
                inp, states[li] = cell(inp, states[li])
        return states

    def decode(
        self,
        init_states: list[tuple[torch.Tensor, torch.Tensor]],
        init_target: torch.Tensor,
        aux_future: torch.Tensor,
        *,
        teacher_forcing_target: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive decode. Always returns absolute predictions.

        init_target: (B, C_tgt, H, W) — last observed target frame (decoder seed).
        aux_future:  (B, T_out, C_aux, H, W) — aux channels per future step.
        teacher_forcing_target: (B, T_out, C_tgt, H, W) absolute targets for TF.

        Residual mode: head outputs delta Δy; y_abs = init_target + Δy is used
        as both the output and the feedback prev_y. This gives the model a
        persistence inductive bias at initialisation.
        """
        B, T_out, _, H, W = aux_future.shape
        states = [(h.clone(), c.clone()) for (h, c) in init_states]
        prev_y = init_target                                               # absolute, (B, C_tgt, H, W)
        outputs = []
        for t in range(T_out):
            inp = torch.cat([prev_y, aux_future[:, t]], dim=1)            # (B, C_tgt+C_aux, H, W)
            for li, cell in enumerate(self.decoder):
                inp, states[li] = cell(inp, states[li])
            raw = self.head(inp)                                           # (B, C_tgt, H, W)
            y_t = (init_target + raw) if self.residual else raw            # absolute
            outputs.append(y_t)

            # Scheduled sampling: feed either true absolute target or own prediction.
            if (
                self.training
                and teacher_forcing_target is not None
                and teacher_forcing_prob > 0.0
                and t < T_out - 1
            ):
                use_tf = torch.rand((), device=y_t.device) < teacher_forcing_prob
                prev_y = teacher_forcing_target[:, t] if use_tf else y_t
            else:
                prev_y = y_t

        return torch.stack(outputs, dim=1)                                 # (B, T_out, C_tgt, H, W)

    def forward(
        self,
        x_seq: torch.Tensor,
        aux_future: torch.Tensor,
        *,
        init_target: torch.Tensor,
        teacher_forcing_target: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        states = self.encode(x_seq)
        return self.decode(
            states,
            init_target,
            aux_future,
            teacher_forcing_target=teacher_forcing_target,
            teacher_forcing_prob=teacher_forcing_prob,
        )
