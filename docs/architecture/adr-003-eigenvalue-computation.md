# ADR-003: Hermitian Eigenvalue Computation

**Status**: Accepted

## Context

Von Neumann entropy, state fidelity, negativity, and other functions require eigenvalues of Hermitian density matrices. The naive approach (extract real part, run Jacobi) fails for matrices with significant imaginary off-diagonal elements.

## Decision

Use the 2n×2n real symmetric embedding: for an n×n Hermitian matrix H, construct the real symmetric matrix M = [[Re(H), -Im(H)], [Im(H), Re(H)]]. The eigenvalues of M are the eigenvalues of H, each appearing twice. Run Jacobi on M, then deduplicate.

## Key Choices

- **Jacobi over QR**: Jacobi is simpler, always converges for symmetric matrices, and works well for small matrices (quantum density matrices are typically 2-16 dimensional). QR is available via hisab bridge for larger systems.
- **In-place rotation**: Eliminated per-iteration matrix clone — O(n²) memory instead of O(n⁴) transient allocations.
- **Sign-aware degenerate case**: When diagonal elements are equal, the rotation angle accounts for the sign of the off-diagonal element.
- **Hisab bridge fallback**: `bridge::eigenvalues_hermitian` uses hisab's QR-based `eigen_hermitian` for consumers who need higher precision or larger matrices.

## Consequences

- Correct for all Hermitian matrices (real and complex)
- 16.5% speedup from in-place Jacobi (68.9ns → 57.5ns for 4×4)
- 2n×2n embedding doubles the matrix size — acceptable for small quantum systems, but the bridge should be preferred for >8 qubits
