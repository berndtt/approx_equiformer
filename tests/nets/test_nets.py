# test_relax_linear_rs.py
import math
import torch
import pytest
from e3nn import o3

from nets import RelaxedLinearRS


# -------------------------- helpers --------------------------
def random_rotation_matrix():
    return o3.rand_matrix()

def apply_rep(ir: o3.Irreps, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the block-diagonal rep D(R) to a batch of features x (..., dim).
    Uses the standard convention v' = D v  ⇒  x' = x @ D^T for row-major batches.
    """
    D = ir.D_from_matrix(R)  # (dim, dim)
    return x @ D.T

def kron_project_block(B: torch.Tensor, mul_out: int, d: int, mul_in: int) -> torch.Tensor:
    """
    Frobenius-orthogonal projection of block B ((mul_out*d) x (mul_in*d))
    onto {A ⊗ I_d}.
    """
    Bv = B.view(mul_out, d, mul_in, d)
    # A*_{ab} = (1/d) * sum_alpha B_{a,alpha,b,alpha}
    A = Bv.diagonal(offset=0, dim1=1, dim2=3).sum(dim=-1) / d  # (mul_out, mul_in)
    I = torch.eye(d, dtype=B.dtype, device=B.device)
    return torch.kron(A, I)


# -------------------------- tests ----------------------------
@pytest.mark.parametrize("batch", [2, 5])
def test_shapes_forward(batch):
    irreps_in = o3.Irreps("3x0e+2x1o+1x2e")
    irreps_out = o3.Irreps("1x0e+4x1o+2x2e")

    layer = RelaxedLinearRS(irreps_in, irreps_out, bias=True)

    x = torch.randn(batch, irreps_in.dim)
    y = layer(x)
    print(y.shape)
    assert y.shape == (batch, irreps_out.dim)


def test_projection_correctness_blocks_and_kron():
    irreps_in = o3.Irreps("2x0e+3x1o+1x2e")
    irreps_out = o3.Irreps("1x0e+2x1o+2x2e")

    layer = RelaxedLinearRS(irreps_in, irreps_out, bias=False)

    W = layer.weight.detach().clone()
    W_eq = layer.equivariant_projection().detach()

    in_slices = layer.irreps_in.slices()
    out_slices = layer.irreps_out.slices()
    in_sig = [(ir.l, ir.p) for (mul, ir) in layer.irreps_in]
    out_sig = [(ir.l, ir.p) for (mul, ir) in layer.irreps_out]
    in_mul = [mul for (mul, ir) in layer.irreps_in]
    out_mul = [mul for (mul, ir) in layer.irreps_out]
    in_dim = [ir.dim for (mul, ir) in layer.irreps_in]
    out_dim = [ir.dim for (mul, ir) in layer.irreps_out]

    # Off-type blocks are zero; on-type blocks equal Kron(A, I_d)
    for j_out, sl_out in enumerate(out_slices):
        for i_in, sl_in in enumerate(in_slices):
            B_eq = W_eq[sl_out, sl_in]
            if out_sig[j_out] != in_sig[i_in]:
                assert torch.allclose(B_eq, torch.zeros_like(B_eq), atol=1e-7, rtol=0)
            else:
                d = out_dim[j_out]
                assert d == in_dim[i_in]
                B = W[sl_out, sl_in]
                B_proj = kron_project_block(B, out_mul[j_out], d, in_mul[i_in])
                assert torch.allclose(B_eq, B_proj, atol=1e-6, rtol=1e-6)


def test_grad_through_penalty():
    irreps = o3.Irreps("2x0e+2x1o")
    layer = RelaxedLinearRS(irreps, irreps, bias=True)

    pen_eq, pen_non = layer.penalty_terms()
    loss = pen_eq + pen_non
    loss.backward()

    assert layer.weight.grad is not None
    if layer.bias is not None:
        # bias does not appear in penalties, so grad may be None; just ensure no crash
        pass


@pytest.mark.parametrize("batch", [4])
def test_equivariance_after_projection(batch):
    irreps_in = o3.Irreps("2x0e+2x1o+1x2e")
    irreps_out = o3.Irreps("1x0e+3x1o+1x2e")
    layer = RelaxedLinearRS(irreps_in, irreps_out, bias=True)

    # Hard-project to the equivariant subspace (and legal bias)
    layer.project_to_equivariant_(project_bias=True)

    x = torch.randn(batch, irreps_in.dim)
    R = random_rotation_matrix()

    x_rot = apply_rep(irreps_in, R, x)
    y = layer(x)
    y_rot = layer(x_rot)

    # Expected equivariant transform on outputs
    y_expected = apply_rep(irreps_out, R, y)

    assert torch.allclose(y_rot, y_expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch", [4])
def test_not_equivariant_before_projection(batch):
    irreps_in = o3.Irreps("2x0e+2x1o+1x2e")
    irreps_out = o3.Irreps("1x0e+3x1o+1x2e")
    layer = RelaxedLinearRS(irreps_in, irreps_out, bias=False)  # random dense map

    x = torch.randn(batch, irreps_in.dim)
    R = random_rotation_matrix()

    x_rot = apply_rep(irreps_in, R, x)
    y = layer(x)
    y_rot = layer(x_rot)

    y_expected = apply_rep(irreps_out, R, y)

    # With high probability, a random dense map is not equivariant
    assert not torch.allclose(y_rot, y_expected, atol=1e-6, rtol=1e-6)


def test_bias_projection_only_scalars():
    irreps_in = o3.Irreps("1x0e+1x1o")
    irreps_out = o3.Irreps("2x0e+1x1o")
    layer = RelaxedLinearRS(irreps_in, irreps_out, bias=True)

    with torch.no_grad():
        layer.bias.uniform_(-0.5, 0.5)

    layer.project_to_equivariant_(project_bias=True)

    # Bias must be zero on non-scalars (l>0) and free on 0e
    scalar_mask = torch.zeros(irreps_out.dim, dtype=torch.bool)
    for (mul, ir), sl in zip(layer.irreps_out, layer.irreps_out.slices()):
        if ir.l == 0 and ir.p == 1:
            scalar_mask[sl] = True
    non_scalar_bias = layer.bias[~scalar_mask]
    assert torch.allclose(non_scalar_bias, torch.zeros_like(non_scalar_bias), atol=1e-7, rtol=0)
