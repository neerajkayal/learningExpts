import torch

def reversal_mean_rms(A: torch.Tensor):
    """
    Simple reversal invariants (not generically discriminative).

    Args:
        A: real tensor of shape (m, d)

    Returns:
        mean_sym: (ceil(m/2), d) symmetric means
        rms_sym:  (ceil(m/2),)   symmetric RMS values
    """
    assert A.ndim == 2
    m, d = A.shape

    # Indices i and -i = m+1-i
    idx = torch.arange((m + 1) // 2, device=A.device)
    rev_idx = m - 1 - idx

    Ai = A[idx]
    Arev = A[rev_idx]

    mean_sym = 0.5 * (Ai + Arev)
    rms_sym = torch.sqrt(
        0.5 * ((Ai ** 2).sum(dim=1) + (Arev ** 2).sum(dim=1))
    )

    return mean_sym, rms_sym


def reversal_cubic_invariants(A: torch.Tensor):
    """
    Cubic reversal-invariant features (generically discriminative).

    Args:
        A: real tensor of shape (m, d)

    Returns:
        B: real tensor of shape (m, d)
           symmetrized bispectrum chain
    """
    assert A.ndim == 2
    m, d = A.shape

    # FFT along sequence dimension
    A_hat = torch.fft.fft(A, dim=0)  # (m, d), complex

    # Reference frequency
    ref = A_hat[1]

    idx = torch.arange(m, device=A.device)
    neg_idx = (-1 - idx) % m

    # Complex bispectrum chain
    B_complex = ref.unsqueeze(0) * A_hat * A_hat[neg_idx]

    # Reversal-invariant symmetrization
    B = B_complex.real

    return B


def reversal_cubic_invariants(A: torch.Tensor):
    """
    Cubic reversal-invariant features (generically discriminative).

    Args:
        A: real tensor of shape (m, d)

    Returns:
        B: real tensor of shape (m, d)
           symmetrized bispectrum chain
    """
    assert A.ndim == 2
    m, d = A.shape

    # FFT along sequence dimension
    A_hat = torch.fft.fft(A, dim=0)  # (m, d), complex

    # Reference frequency
    ref = A_hat[1]

    idx = torch.arange(m, device=A.device)
    neg_idx = (-1 - idx) % m

    # Complex bispectrum chain
    B_complex = ref.unsqueeze(0) * A_hat * A_hat[neg_idx]

    # Reversal-invariant symmetrization
    B = B_complex.real

    return B


def compute_all_reversal_invariants(A: torch.Tensor):
    """
    Convenience wrapper.

    Returns:
        mean_sym: (ceil(m/2), d)
        rms_sym:  (ceil(m/2),)
        B:        (m, d) cubic generically-discriminative invariants
    """
    mean_sym, rms_sym = reversal_mean_rms(A)
    B = reversal_cubic_invariants(A)
    return mean_sym, rms_sym, B
