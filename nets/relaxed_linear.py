import math
from typing import Tuple

import torch
import torch.nn as nn
from e3nn import o3


class RelaxedLinearRS(nn.Module):
    """
    Non-equivariant li  near layer with the same interface as LinearRS:

        RelaxLinearRS(irreps_in, irreps_out, bias=True, rescale=True)

    • Learns a full dense map that can mix irreps arbitrarily.
    • Provides projection onto the equivariant subspace (block structure A ⊗ I_d).
    • Provides Frobenius-norm penalties (‖W_eq‖_F, ‖W - W_eq‖_F).

    Shapes:
        x: (..., irreps_in.dim)
        y: (..., irreps_out.dim)
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps,
                 bias: bool = True, rescale: bool = True):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.rescale = rescale
        self.use_bias = bias

        self.C_in = self.irreps_in.dim
        self.C_out = self.irreps_out.dim

        # Full dense weight (non-equivariant)
        W = torch.empty(self.C_out, self.C_in)
        # torch.nn.Linear-style init
        bound = 1.0 / math.sqrt(self.C_in) if self.rescale else 1.0
        nn.init.uniform_(W, -bound, bound)
        self.weight = nn.Parameter(W)

        if self.use_bias:
            b = torch.zeros(self.C_out)
            # same scaling as nn.Linear
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter("bias", None)

        # Precompute slices and metadata for equivariant projection
        self._in_slices = self.irreps_in.slices()
        self._out_slices = self.irreps_out.slices()
        # (l, p) signatures per block
        self._in_sig = [(ir.l, ir.p) for (mul, ir) in self.irreps_in]
        self._out_sig = [(ir.l, ir.p) for (mul, ir) in self.irreps_out]
        self._in_mul = [mul for (mul, ir) in self.irreps_in]
        self._out_mul = [mul for (mul, ir) in self.irreps_out]
        self._in_dim = [ir.dim for (mul, ir) in self.irreps_in]
        self._out_dim = [ir.dim for (mul, ir) in self.irreps_out]

        # Indices of scalar-even (0e) outputs for bias projection (if desired)
        self._scalar_out_mask = torch.zeros(self.C_out, dtype=torch.bool)
        for idx, ((mul, ir), sl) in enumerate(zip(self.irreps_out, self._out_slices)):
            if ir.l == 0 and ir.p == 1:  # 0e
                self._scalar_out_mask[sl] = True

    # ------------------------------------------------------------------ #
    # Forward: y = x W^T + b
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C_in)
        y = torch.einsum("...i,oi->...o", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    # ------------------------------------------------------------------ #
    # Equivariant projection utilities
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def project_to_equivariant_(self, project_bias: bool = True) -> None:
        """
        In-place: W ← Proj_eq(W). Optionally project bias to 0e only.
        """
        W_eq = self.equivariant_projection()
        self.weight.copy_(W_eq)

        if project_bias and (self.bias is not None):
            # True equivariance only allows bias on 0e outputs
            b = self.bias
            b = b * self._scalar_out_mask.to(b.dtype)
            self.bias.copy_(b)

    def equivariant_projection(self) -> torch.Tensor:
        """
        Return W_eq, the orthogonal projection of W onto the space of
        equivariant linear maps:
            block_{(l,p)->(l,p)} = A ⊗ I_d,  zero otherwise.

        The projection (Frobenius-optimal) for a block B of shape
        (m_out * d) x (m_in * d) is:
            A* = (1/d) * sum_{α=1..d} B[:, α, :, α]
            Proj(B) = A* ⊗ I_d
        """
        W = self.weight
        W_eq = torch.zeros_like(W)

        for j_out, ((mul_out, ir_out), sl_out, d_out) in enumerate(
            zip(self.irreps_out, self._out_slices, self._out_dim)
        ):
            for i_in, ((mul_in, ir_in), sl_in, d_in) in enumerate(
                zip(self.irreps_in, self._in_slices, self._in_dim)
            ):
                # Only same irrep type contributes (same l and parity)
                if (ir_out.l, ir_out.p) != (ir_in.l, ir_in.p):
                    continue

                # d_out == d_in == d
                d = d_out
                assert d_out == d_in, "Mismatched irrep dimensions for same (l,p)."

                # Extract block and reshape to (m_out, d, m_in, d)
                B = W[sl_out, sl_in]
                Bv = B.view(mul_out, d, mul_in, d)

                # A*_{ab} = (1/d) * sum_alpha B_{a,alpha,b,alpha}
                A = Bv.diagonal(offset=0, dim1=1, dim2=3).sum(dim=-1) / d  # (m_out, m_in)

                # Proj(B) = A ⊗ I_d
                # Use kronecker for clarity
                I = torch.eye(d, dtype=W.dtype, device=W.device)
                B_proj = torch.kron(A, I)  # (m_out*d, m_in*d)

                W_eq[sl_out, sl_in] = B_proj

        return W_eq

    def penalty_terms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (‖W_eq‖_F, ‖W - W_eq‖_F) for use in regularization.
        """
        W_eq = self.equivariant_projection()
        W_non = self.weight - W_eq
        return W_eq.norm(), W_non.norm()
