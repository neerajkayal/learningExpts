import torch

def cyclic_invariants(A: torch.Tensor):
    """
    A: real tensor of shape (m, d)
       m = sequence length
       d = ambient dimension

    Returns:
        M : real tensor of shape (m,)
            quadratic (power spectrum) invariants
        B : complex tensor of shape (m, d)
            cubic (bispectrum chain) invariants
    """
    assert A.ndim == 2
    m, d = A.shape

    # FFT along the sequence dimension
    # Result: (m, d) complex
    A_hat = torch.fft.fft(A, dim=0) # shape (m, d), complex valued

    # -----------------------------
    # Quadratic invariants (M)
    # -----------------------------
    # Power spectrum: ||A_hat[ℓ]||^2
    M = (A_hat.conj() * A_hat).real.sum(dim=1)  # shape (m,), real valued

    # -----------------------------
    # Cubic invariants (B)
    # -----------------------------
    # Reference frequency ℓ = 1
    ref = A_hat[1]  # shape (d,), complex valued

    # Frequencies 0,...,m-1
    idx = torch.arange(m, device=A.device) # shape (m,), long

    # Compute -(1 + ℓ) mod m
    neg_idx = (-1 - idx) % m  # shape (m,), long

    # Bispectrum chain
    # Elementwise product in C^d
    B = ref.unsqueeze(0) * A_hat * A_hat[neg_idx]  # shape (m, d), complex valued
    return M, B


def cyclic_shift(A: torch.Tensor, k: int):
    """
    Circularly shift A by k positions along the first dimension.
    """
    return torch.roll(A, shifts=-k, dims=0)

def test_cyclic_invariance(
    m=16,
    d=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
    atol=1e-5,
    rtol=1e-5,
):
    torch.manual_seed(0)

    # Random input
    A = torch.randn(m, d, device=device, dtype=dtype)

    # Original invariants
    M, B = cyclic_invariants(A)

    # Test multiple random shifts
    for k in range(1, m):
        A_shift = cyclic_shift(A, k)
        M_shift, B_shift = cyclic_invariants(A_shift)

        if not torch.allclose(M, M_shift, atol=atol, rtol=rtol):
            raise AssertionError(f"M not invariant under shift {k}")

        if not torch.allclose(B, B_shift, atol=atol, rtol=rtol):
            raise AssertionError(f"B not invariant under shift {k}")

        print(f"Cyclic invariance test passed OK for shift {k}")
    return

if __name__ == "__main__":
    test_cyclic_invariance()