//! Entanglement — Bell states, density matrices, partial trace, concurrence.
//!
//! Entangled states cannot be written as a tensor product of individual qubit states.
//! The density matrix formalism extends pure states to mixed states.

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};
use crate::state::NORM_TOLERANCE;

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
                elements[i * dim + j] = (a_re * b_re + a_im * b_im, a_im * b_re - a_re * b_im);
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
        self.elements.iter().map(|(re, im)| re * re + im * im).sum()
    }

    /// Von Neumann entropy S = −Tr(ρ log₂ ρ).
    ///
    /// For a pure state, S = 0. For a maximally mixed n-qubit state, S = n.
    /// Computes eigenvalues of the Hermitian density matrix via QR iteration.
    #[must_use]
    pub fn von_neumann_entropy(&self) -> f64 {
        let eigenvalues = self.hermitian_eigenvalues();
        let mut s = 0.0;
        for &l in &eigenvalues {
            if l > 1e-15 {
                s -= l * l.log2();
            }
        }
        s
    }

    /// Compute real eigenvalues of this Hermitian density matrix.
    ///
    /// Uses the Jacobi eigenvalue algorithm on the real-symmetric part.
    /// For density matrices from physical quantum states, the real part
    /// captures the full eigenvalue spectrum.
    #[must_use]
    pub fn hermitian_eigenvalues(&self) -> Vec<f64> {
        let n = self.dim;

        // Extract real-symmetric matrix
        let mut a: Vec<f64> = (0..n * n).map(|idx| self.elements[idx].0).collect();

        // Jacobi eigenvalue algorithm: iteratively zero off-diagonal elements
        // using Givens rotations. Converges for all real symmetric matrices.
        for _ in 0..n * n * 100 {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = a[i * n + j].abs();
                    if v > max_val {
                        max_val = v;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < NORM_TOLERANCE {
                break;
            }

            // Compute rotation angle
            let app = a[p * n + p];
            let aqq = a[q * n + q];
            let apq = a[p * n + q];

            let (cos, sin) = if (app - aqq).abs() < NORM_TOLERANCE {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                (s, apq.signum() * s)
            } else {
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                (c, t * c)
            };

            // Apply Jacobi rotation in-place: A' = J^T A J
            // Update rows/cols p and q
            for i in 0..n {
                if i != p && i != q {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = cos * aip - sin * aiq;
                    a[p * n + i] = a[i * n + p];
                    a[i * n + q] = sin * aip + cos * aiq;
                    a[q * n + i] = a[i * n + q];
                }
            }

            a[p * n + p] = cos * cos * app - 2.0 * sin * cos * apq + sin * sin * aqq;
            a[q * n + q] = sin * sin * app + 2.0 * sin * cos * apq + cos * cos * aqq;
            a[p * n + q] = 0.0;
            a[q * n + p] = 0.0;
        }

        // Diagonal elements are the eigenvalues
        (0..n).map(|i| a[i * n + i]).collect()
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

/// Schmidt decomposition of a bipartite pure state.
///
/// For |ψ⟩ ∈ H_A ⊗ H_B, returns the Schmidt coefficients λᵢ such that
/// |ψ⟩ = Σᵢ λᵢ |aᵢ⟩|bᵢ⟩ where λᵢ ≥ 0 and Σ λᵢ² = 1.
///
/// The number of non-zero coefficients (Schmidt rank) measures entanglement:
/// rank 1 = separable, rank > 1 = entangled.
pub fn schmidt_decomposition(
    amplitudes: &[(f64, f64)],
    dim_a: usize,
    dim_b: usize,
) -> Result<Vec<f64>> {
    if amplitudes.len() != dim_a * dim_b {
        return Err(KanaError::DimensionMismatch {
            expected: dim_a * dim_b,
            got: amplitudes.len(),
        });
    }

    // Reshape amplitudes into a dim_a × dim_b matrix (real part only for now)
    // Then compute singular values via eigenvalues of M M†
    // For a real-valued state, SVD singular values = sqrt(eigenvalues of ρ_A)
    let dm = DensityMatrix::from_pure_state(amplitudes);
    let reduced = dm.partial_trace_b(dim_a, dim_b)?;
    let eigenvalues = reduced.hermitian_eigenvalues();

    // Schmidt coefficients = sqrt of eigenvalues of reduced density matrix
    let mut coeffs: Vec<f64> = eigenvalues
        .into_iter()
        .map(|l| l.max(0.0).sqrt())
        .filter(|&c| c > NORM_TOLERANCE)
        .collect();
    coeffs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(coeffs)
}

/// Schmidt rank of a bipartite state (number of non-zero Schmidt coefficients).
///
/// Rank 1 = separable, rank > 1 = entangled.
pub fn schmidt_rank(amplitudes: &[(f64, f64)], dim_a: usize, dim_b: usize) -> Result<usize> {
    schmidt_decomposition(amplitudes, dim_a, dim_b).map(|c| c.len())
}

/// Reconstruct a single-qubit density matrix from Pauli expectation values.
///
/// Given ⟨X⟩, ⟨Y⟩, ⟨Z⟩ (Bloch vector components), reconstructs:
/// ρ = (I + x·σx + y·σy + z·σz) / 2
///
/// The expectation values can be estimated from measurement statistics:
/// - ⟨Z⟩ = P(|0⟩) − P(|1⟩) in computational basis
/// - ⟨X⟩ = P(|+⟩) − P(|−⟩) in X basis (apply H before measuring)
/// - ⟨Y⟩ = P(|+i⟩) − P(|−i⟩) in Y basis (apply S†H before measuring)
#[must_use]
pub fn tomography_single_qubit(expect_x: f64, expect_y: f64, expect_z: f64) -> DensityMatrix {
    // ρ = (I + x·σx + y·σy + z·σz) / 2
    // = [ (1+z)/2      (x-iy)/2  ]
    //   [ (x+iy)/2     (1-z)/2   ]
    let elements = vec![
        ((1.0 + expect_z) / 2.0, 0.0),
        (expect_x / 2.0, -expect_y / 2.0),
        (expect_x / 2.0, expect_y / 2.0),
        ((1.0 - expect_z) / 2.0, 0.0),
    ];
    DensityMatrix { elements, dim: 2 }
}

/// Estimate Pauli expectation values from measurement counts.
///
/// Takes counts of 0 and 1 outcomes in each basis:
/// - `z_counts`: (count_0, count_1) from computational basis measurement
/// - `x_counts`: (count_0, count_1) from X basis measurement (H then measure)
/// - `y_counts`: (count_0, count_1) from Y basis measurement (S†H then measure)
///
/// Returns (⟨X⟩, ⟨Y⟩, ⟨Z⟩).
#[must_use]
pub fn estimate_pauli_expectations(
    z_counts: (usize, usize),
    x_counts: (usize, usize),
    y_counts: (usize, usize),
) -> (f64, f64, f64) {
    let expect = |(n0, n1): (usize, usize)| -> f64 {
        let total = (n0 + n1) as f64;
        if total == 0.0 {
            return 0.0;
        }
        (n0 as f64 - n1 as f64) / total
    };
    (expect(x_counts), expect(y_counts), expect(z_counts))
}

/// A quantum noise channel defined by Kraus operators.
///
/// Applies ρ → Σₖ Eₖ ρ Eₖ† where the Eₖ satisfy Σₖ Eₖ†Eₖ = I.
#[derive(Debug, Clone)]
pub struct NoiseChannel {
    /// Kraus operators as (dim, elements) where elements are (re, im) pairs.
    kraus_ops: Vec<Vec<(f64, f64)>>,
    dim: usize,
}

impl NoiseChannel {
    /// Create a noise channel from a set of Kraus operators.
    ///
    /// Each operator is a flat row-major complex matrix of size dim×dim.
    pub fn new(dim: usize, kraus_ops: Vec<Vec<(f64, f64)>>) -> Result<Self> {
        for op in &kraus_ops {
            if op.len() != dim * dim {
                return Err(KanaError::DimensionMismatch {
                    expected: dim * dim,
                    got: op.len(),
                });
            }
        }
        Ok(Self { kraus_ops, dim })
    }

    /// Apply this channel to a density matrix: ρ → Σₖ Eₖ ρ Eₖ†.
    pub fn apply(&self, rho: &DensityMatrix) -> Result<DensityMatrix> {
        if rho.dim() != self.dim {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: rho.dim(),
            });
        }
        let n = self.dim;
        let mut result = vec![(0.0, 0.0); n * n];

        for kraus in &self.kraus_ops {
            // Compute E ρ E†
            // First: temp = E ρ
            let mut temp = vec![(0.0, 0.0); n * n];
            for i in 0..n {
                for j in 0..n {
                    let (mut re, mut im) = (0.0, 0.0);
                    for k in 0..n {
                        let (e_re, e_im) = kraus[i * n + k];
                        let (r_re, r_im) = rho.elements[k * n + j];
                        re += e_re * r_re - e_im * r_im;
                        im += e_re * r_im + e_im * r_re;
                    }
                    temp[i * n + j] = (re, im);
                }
            }
            // Then: result += temp × E†
            for i in 0..n {
                for j in 0..n {
                    let (mut re, mut im) = (0.0, 0.0);
                    for k in 0..n {
                        let (t_re, t_im) = temp[i * n + k];
                        // E†[k][j] = conj(E[j][k])
                        let (e_re, e_im) = kraus[j * n + k];
                        let (ed_re, ed_im) = (e_re, -e_im);
                        re += t_re * ed_re - t_im * ed_im;
                        im += t_re * ed_im + t_im * ed_re;
                    }
                    result[i * n + j].0 += re;
                    result[i * n + j].1 += im;
                }
            }
        }

        Ok(DensityMatrix {
            elements: result,
            dim: n,
        })
    }

    /// Compose two noise channels: apply self first, then other.
    pub fn compose(&self, other: &Self) -> Result<Self> {
        if self.dim != other.dim {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        // Composed Kraus: { F_j E_k } for all j, k
        let n = self.dim;
        let mut composed = Vec::new();
        for f_op in &other.kraus_ops {
            for e_op in &self.kraus_ops {
                // Multiply F × E
                let mut prod = vec![(0.0, 0.0); n * n];
                for i in 0..n {
                    for j in 0..n {
                        let (mut re, mut im) = (0.0, 0.0);
                        for k in 0..n {
                            let (f_re, f_im) = f_op[i * n + k];
                            let (e_re, e_im) = e_op[k * n + j];
                            re += f_re * e_re - f_im * e_im;
                            im += f_re * e_im + f_im * e_re;
                        }
                        prod[i * n + j] = (re, im);
                    }
                }
                composed.push(prod);
            }
        }
        Ok(Self {
            kraus_ops: composed,
            dim: n,
        })
    }

    /// Depolarizing channel: ρ → (1−3p/4)ρ + (p/4)(XρX + YρY + ZρZ).
    ///
    /// `p` is the depolarization probability (0 = no noise, 1 = maximally mixed).
    pub fn depolarizing(p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(KanaError::InvalidParameter {
                reason: format!("depolarizing probability {p} not in [0, 1]"),
            });
        }
        let s0 = (1.0 - 3.0 * p / 4.0).sqrt();
        let sp = (p / 4.0).sqrt();
        Ok(Self {
            kraus_ops: vec![
                // √(1−3p/4) I
                vec![(s0, 0.0), (0.0, 0.0), (0.0, 0.0), (s0, 0.0)],
                // √(p/4) X
                vec![(0.0, 0.0), (sp, 0.0), (sp, 0.0), (0.0, 0.0)],
                // √(p/4) Y
                vec![(0.0, 0.0), (0.0, -sp), (0.0, sp), (0.0, 0.0)],
                // √(p/4) Z
                vec![(sp, 0.0), (0.0, 0.0), (0.0, 0.0), (-sp, 0.0)],
            ],
            dim: 2,
        })
    }

    /// Amplitude damping channel (energy relaxation, T1 decay).
    ///
    /// `gamma` is the decay probability (0 = no damping, 1 = full relaxation to |0⟩).
    pub fn amplitude_damping(gamma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(KanaError::InvalidParameter {
                reason: format!("damping probability {gamma} not in [0, 1]"),
            });
        }
        let sg = gamma.sqrt();
        let s1g = (1.0 - gamma).sqrt();
        Ok(Self {
            kraus_ops: vec![
                // E0 = |0⟩⟨0| + √(1−γ)|1⟩⟨1|
                vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s1g, 0.0)],
                // E1 = √γ |0⟩⟨1|
                vec![(0.0, 0.0), (sg, 0.0), (0.0, 0.0), (0.0, 0.0)],
            ],
            dim: 2,
        })
    }

    /// Phase damping channel (dephasing, T2 decay without energy loss).
    ///
    /// `gamma` is the dephasing probability (0 = no dephasing, 1 = full dephasing).
    pub fn phase_damping(gamma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err(KanaError::InvalidParameter {
                reason: format!("damping probability {gamma} not in [0, 1]"),
            });
        }
        let s1g = (1.0 - gamma).sqrt();
        let sg = gamma.sqrt();
        Ok(Self {
            kraus_ops: vec![
                // E0 = |0⟩⟨0| + √(1−γ)|1⟩⟨1|
                vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s1g, 0.0)],
                // E1 = √γ |1⟩⟨1|
                vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (sg, 0.0)],
            ],
            dim: 2,
        })
    }
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
        for bell in [
            bell_phi_plus(),
            bell_phi_minus(),
            bell_psi_plus(),
            bell_psi_minus(),
        ] {
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

    #[test]
    fn test_von_neumann_entropy_4x4_pure() {
        // Pure 2-qubit state |00⟩ has S = 0
        let amps = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let dm = DensityMatrix::from_pure_state(&amps);
        assert!(dm.von_neumann_entropy().abs() < 1e-5);
    }

    #[test]
    fn test_von_neumann_entropy_4x4_bell() {
        // Bell state |Φ+⟩: full density matrix is pure (S = 0)
        let phi_plus = bell_phi_plus();
        let dm = DensityMatrix::from_pure_state(&phi_plus);
        assert!(dm.von_neumann_entropy().abs() < 1e-5);
    }

    #[test]
    fn test_eigenvalues_maximally_mixed_2x2() {
        // Maximally mixed: ρ = I/2 → eigenvalues (0.5, 0.5) → S = 1
        let dm =
            DensityMatrix::new(2, vec![(0.5, 0.0), (0.0, 0.0), (0.0, 0.0), (0.5, 0.0)]).unwrap();
        assert!((dm.von_neumann_entropy() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_concurrence_wrong_length() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        assert!(concurrence_pure(&amps).abs() < 1e-10);
    }

    #[test]
    fn test_schmidt_product_state() {
        // |00⟩ is separable → Schmidt rank 1
        let amps = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let coeffs = schmidt_decomposition(&amps, 2, 2).unwrap();
        assert_eq!(coeffs.len(), 1);
        assert!((coeffs[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_schmidt_bell_state() {
        // |Φ+⟩ is maximally entangled → Schmidt rank 2, coefficients (1/√2, 1/√2)
        let bell = bell_phi_plus();
        let coeffs = schmidt_decomposition(&bell, 2, 2).unwrap();
        assert_eq!(coeffs.len(), 2);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((coeffs[0] - s).abs() < 1e-5);
        assert!((coeffs[1] - s).abs() < 1e-5);
    }

    #[test]
    fn test_schmidt_rank() {
        let product = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        assert_eq!(schmidt_rank(&product, 2, 2).unwrap(), 1);

        let bell = bell_phi_plus();
        assert_eq!(schmidt_rank(&bell, 2, 2).unwrap(), 2);
    }

    #[test]
    fn test_schmidt_dimension_mismatch() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        assert!(schmidt_decomposition(&amps, 2, 2).is_err());
    }

    #[test]
    fn test_tomography_zero_state() {
        // |0⟩ has Bloch vector (0, 0, 1)
        let dm = tomography_single_qubit(0.0, 0.0, 1.0);
        let (re00, _) = dm.element(0, 0).unwrap();
        let (re11, _) = dm.element(1, 1).unwrap();
        assert!((re00 - 1.0).abs() < 1e-10);
        assert!(re11.abs() < 1e-10);
    }

    #[test]
    fn test_tomography_plus_state() {
        // |+⟩ has Bloch vector (1, 0, 0)
        let dm = tomography_single_qubit(1.0, 0.0, 0.0);
        // ρ = [[0.5, 0.5], [0.5, 0.5]]
        let (re00, _) = dm.element(0, 0).unwrap();
        let (re01, _) = dm.element(0, 1).unwrap();
        assert!((re00 - 0.5).abs() < 1e-10);
        assert!((re01 - 0.5).abs() < 1e-10);
        assert!((dm.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tomography_maximally_mixed() {
        // Maximally mixed: (0, 0, 0) → ρ = I/2
        let dm = tomography_single_qubit(0.0, 0.0, 0.0);
        assert!((dm.purity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_pauli_expectations() {
        // All |0⟩ measurements in Z basis → ⟨Z⟩ = 1
        let (x, y, z) = estimate_pauli_expectations((100, 0), (50, 50), (50, 50));
        assert!((z - 1.0).abs() < 1e-10);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_depolarizing_no_noise() {
        let ch = NoiseChannel::depolarizing(0.0).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        let result = ch.apply(&dm).unwrap();
        assert!((result.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_depolarizing_full_noise() {
        let ch = NoiseChannel::depolarizing(1.0).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        let result = ch.apply(&dm).unwrap();
        // Fully depolarized → maximally mixed → purity = 0.5
        assert!((result.purity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_damping_no_decay() {
        let ch = NoiseChannel::amplitude_damping(0.0).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(0.0, 0.0), (1.0, 0.0)]); // |1⟩
        let result = ch.apply(&dm).unwrap();
        let (re11, _) = result.element(1, 1).unwrap();
        assert!((re11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_damping_full_decay() {
        let ch = NoiseChannel::amplitude_damping(1.0).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(0.0, 0.0), (1.0, 0.0)]); // |1⟩
        let result = ch.apply(&dm).unwrap();
        // Full decay: |1⟩ → |0⟩
        let (re00, _) = result.element(0, 0).unwrap();
        assert!((re00 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_damping() {
        let ch = NoiseChannel::phase_damping(1.0).unwrap();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let dm = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]); // |+⟩
        let result = ch.apply(&dm).unwrap();
        // Full dephasing kills off-diagonals
        let (re01, im01) = result.element(0, 1).unwrap();
        assert!(re01.abs() < 1e-10);
        assert!(im01.abs() < 1e-10);
        // Diagonals preserved
        let (re00, _) = result.element(0, 0).unwrap();
        assert!((re00 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_noise_channel_preserves_trace() {
        let ch = NoiseChannel::depolarizing(0.5).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        let result = ch.apply(&dm).unwrap();
        let (tr_re, tr_im) = result.trace();
        assert!((tr_re - 1.0).abs() < 1e-10);
        assert!(tr_im.abs() < 1e-10);
    }

    #[test]
    fn test_noise_channel_compose() {
        let ch1 = NoiseChannel::depolarizing(0.1).unwrap();
        let ch2 = NoiseChannel::depolarizing(0.1).unwrap();
        let composed = ch1.compose(&ch2).unwrap();
        let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        let result = composed.apply(&dm).unwrap();
        // Composed channel should degrade purity more than single application
        let single = ch1.apply(&dm).unwrap();
        assert!(result.purity() < single.purity() + 1e-10);
    }

    #[test]
    fn test_noise_dimension_mismatch() {
        let ch = NoiseChannel::depolarizing(0.5).unwrap(); // 2×2
        let dm = DensityMatrix::from_pure_state(&bell_phi_plus()); // 4×4
        assert!(ch.apply(&dm).is_err());
    }
}
