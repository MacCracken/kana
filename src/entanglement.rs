//! Entanglement — Bell states, density matrices, partial trace, concurrence.
//!
//! Entangled states cannot be written as a tensor product of individual qubit states.
//! The density matrix formalism extends pure states to mixed states.

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};

/// Density matrix ρ for a quantum system.
///
/// For pure states: ρ = |ψ⟩⟨ψ|
/// For mixed states: ρ = Σ pᵢ |ψᵢ⟩⟨ψᵢ|
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMatrix {
    /// Matrix elements in row-major order as (re, im) pairs.
    elements: Vec<(f64, f64)>,
    /// Dimension of the density matrix.
    dim: usize,
}

impl DensityMatrix {
    /// Create a density matrix from raw elements.
    pub fn new(dim: usize, elements: Vec<(f64, f64)>) -> Result<Self> {
        if elements.len() != dim * dim {
            return Err(KanaError::DimensionMismatch {
                expected: dim * dim,
                got: elements.len(),
            });
        }
        Ok(Self { elements, dim })
    }

    /// Create a pure-state density matrix from amplitudes: ρ = |ψ⟩⟨ψ|.
    #[must_use]
    pub fn from_pure_state(amplitudes: &[(f64, f64)]) -> Self {
        let dim = amplitudes.len();
        let mut elements = vec![(0.0, 0.0); dim * dim];
        for i in 0..dim {
            let (a_re, a_im) = amplitudes[i];
            for j in 0..dim {
                let (b_re, b_im) = amplitudes[j];
                // |ψ⟩⟨ψ| element = αᵢ × conj(αⱼ)
                elements[i * dim + j] = (
                    a_re * b_re + a_im * b_im,
                    a_im * b_re - a_re * b_im,
                );
            }
        }
        Self { elements, dim }
    }

    /// Dimension of the density matrix.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get element at (row, col).
    #[inline]
    #[must_use]
    pub fn element(&self, row: usize, col: usize) -> Option<(f64, f64)> {
        if row < self.dim && col < self.dim {
            Some(self.elements[row * self.dim + col])
        } else {
            None
        }
    }

    /// Trace of the density matrix (should be 1.0 for valid states).
    #[must_use]
    pub fn trace(&self) -> (f64, f64) {
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..self.dim {
            let (r, m) = self.elements[i * self.dim + i];
            re += r;
            im += m;
        }
        (re, im)
    }

    /// Purity Tr(ρ²): 1.0 for pure states, < 1.0 for mixed states.
    #[must_use]
    pub fn purity(&self) -> f64 {
        // Tr(ρ²) = Σᵢⱼ |ρᵢⱼ|²
        self.elements
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum()
    }

    /// Von Neumann entropy S = −Tr(ρ log₂ ρ).
    ///
    /// For a pure state, S = 0. For a maximally mixed n-qubit state, S = n.
    /// Uses eigenvalue approximation via purity for 2×2 matrices.
    #[must_use]
    pub fn von_neumann_entropy(&self) -> f64 {
        if self.dim == 2 {
            let p = self.purity();
            // For 2×2: eigenvalues λ± = (1 ± √(2p-1))/2
            let disc = (2.0 * p - 1.0).max(0.0).sqrt();
            let l1 = (1.0 + disc) / 2.0;
            let l2 = (1.0 - disc) / 2.0;
            let mut s = 0.0;
            if l1 > 1e-15 {
                s -= l1 * l1.log2();
            }
            if l2 > 1e-15 {
                s -= l2 * l2.log2();
            }
            s
        } else {
            // For larger systems, we'd need eigendecomposition from hisab
            // Placeholder: estimate from purity
            let p = self.purity();
            if (p - 1.0).abs() < 1e-10 {
                0.0
            } else {
                // Lower bound estimate
                (1.0 - p) * (self.dim as f64).log2()
            }
        }
    }

    /// Partial trace over subsystem B for a bipartite system A⊗B.
    ///
    /// Returns the reduced density matrix for subsystem A.
    pub fn partial_trace_b(&self, dim_a: usize, dim_b: usize) -> Result<Self> {
        if dim_a * dim_b != self.dim {
            return Err(KanaError::IncompatibleSubsystems);
        }
        let mut reduced = vec![(0.0, 0.0); dim_a * dim_a];
        for i in 0..dim_a {
            for j in 0..dim_a {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..dim_b {
                    let row = i * dim_b + k;
                    let col = j * dim_b + k;
                    let (r, m) = self.elements[row * self.dim + col];
                    re += r;
                    im += m;
                }
                reduced[i * dim_a + j] = (re, im);
            }
        }
        Ok(Self {
            elements: reduced,
            dim: dim_a,
        })
    }
}

/// Create the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.
#[must_use]
pub fn bell_phi_plus() -> Vec<(f64, f64)> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![(s, 0.0), (0.0, 0.0), (0.0, 0.0), (s, 0.0)]
}

/// Create the Bell state |Φ−⟩ = (|00⟩ − |11⟩)/√2.
#[must_use]
pub fn bell_phi_minus() -> Vec<(f64, f64)> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![(s, 0.0), (0.0, 0.0), (0.0, 0.0), (-s, 0.0)]
}

/// Create the Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2.
#[must_use]
pub fn bell_psi_plus() -> Vec<(f64, f64)> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![(0.0, 0.0), (s, 0.0), (s, 0.0), (0.0, 0.0)]
}

/// Create the Bell state |Ψ−⟩ = (|01⟩ − |10⟩)/√2.
#[must_use]
pub fn bell_psi_minus() -> Vec<(f64, f64)> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    vec![(0.0, 0.0), (s, 0.0), (-s, 0.0), (0.0, 0.0)]
}

/// Concurrence for a 2-qubit density matrix.
///
/// C = max(0, λ₁ − λ₂ − λ₃ − λ₄) where λᵢ are eigenvalues of
/// √(√ρ ρ̃ √ρ) in decreasing order, and ρ̃ = (σy⊗σy)ρ*(σy⊗σy).
///
/// For pure states: C = 2|ad - bc| where |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩.
#[must_use]
pub fn concurrence_pure(amplitudes: &[(f64, f64)]) -> f64 {
    if amplitudes.len() != 4 {
        return 0.0;
    }
    let (a_re, a_im) = amplitudes[0]; // |00⟩
    let (b_re, b_im) = amplitudes[1]; // |01⟩
    let (c_re, c_im) = amplitudes[2]; // |10⟩
    let (d_re, d_im) = amplitudes[3]; // |11⟩

    // ad - bc (complex multiplication)
    let prod_re = (a_re * d_re - a_im * d_im) - (b_re * c_re - b_im * c_im);
    let prod_im = (a_re * d_im + a_im * d_re) - (b_re * c_im + b_im * c_re);

    2.0 * (prod_re * prod_re + prod_im * prod_im).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_state_density_matrix() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        let dm = DensityMatrix::from_pure_state(&amps);
        let (tr_re, tr_im) = dm.trace();
        assert!((tr_re - 1.0).abs() < 1e-10);
        assert!(tr_im.abs() < 1e-10);
    }

    #[test]
    fn test_pure_state_purity() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let amps = vec![(s, 0.0), (s, 0.0)];
        let dm = DensityMatrix::from_pure_state(&amps);
        assert!((dm.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_entanglement() {
        let phi_plus = bell_phi_plus();
        let c = concurrence_pure(&phi_plus);
        assert!((c - 1.0).abs() < 1e-10); // maximally entangled
    }

    #[test]
    fn test_product_state_no_entanglement() {
        // |00⟩ = not entangled
        let product = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let c = concurrence_pure(&product);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn test_bell_states_maximally_entangled() {
        for bell in [bell_phi_plus(), bell_phi_minus(), bell_psi_plus(), bell_psi_minus()] {
            let c = concurrence_pure(&bell);
            assert!((c - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_partial_trace() {
        let phi_plus = bell_phi_plus();
        let dm = DensityMatrix::from_pure_state(&phi_plus);
        let reduced = dm.partial_trace_b(2, 2).unwrap();
        assert_eq!(reduced.dim(), 2);
        // Reduced state of Bell state is maximally mixed: ρ_A = I/2
        let (re_00, _) = reduced.element(0, 0).unwrap();
        let (re_11, _) = reduced.element(1, 1).unwrap();
        assert!((re_00 - 0.5).abs() < 1e-10);
        assert!((re_11 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy_pure() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        let dm = DensityMatrix::from_pure_state(&amps);
        assert!(dm.von_neumann_entropy().abs() < 1e-10);
    }

    #[test]
    fn test_von_neumann_entropy_mixed() {
        let phi_plus = bell_phi_plus();
        let dm = DensityMatrix::from_pure_state(&phi_plus);
        let reduced = dm.partial_trace_b(2, 2).unwrap();
        // Maximally mixed 2×2 → S = 1 bit
        assert!((reduced.von_neumann_entropy() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_incompatible_partial_trace() {
        let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        assert!(dm.partial_trace_b(3, 2).is_err());
    }
}
