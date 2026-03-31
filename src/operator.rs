//! Quantum operators — unitary operators, observables, Pauli matrices, measurement.
//!
//! Operators act on quantum states via matrix-vector multiplication.
//! Unitary operators preserve the norm: U†U = I.

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};
use crate::state::StateVector;

/// A quantum operator represented as a dense complex matrix.
///
/// Stored in row-major order as (re, im) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operator {
    /// Matrix elements in row-major order.
    elements: Vec<(f64, f64)>,
    /// Dimension of the operator (n×n matrix).
    dim: usize,
}

impl Operator {
    /// Create an operator from a flat row-major complex matrix.
    pub fn new(dim: usize, elements: Vec<(f64, f64)>) -> Result<Self> {
        if elements.len() != dim * dim {
            return Err(KanaError::DimensionMismatch {
                expected: dim * dim,
                got: elements.len(),
            });
        }
        Ok(Self { elements, dim })
    }

    /// Identity operator for n×n.
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        let mut elements = vec![(0.0, 0.0); dim * dim];
        for i in 0..dim {
            elements[i * dim + i] = (1.0, 0.0);
        }
        Self { elements, dim }
    }

    /// Pauli-X (NOT gate): |0⟩↔|1⟩.
    #[must_use]
    pub fn pauli_x() -> Self {
        Self {
            elements: vec![
                (0.0, 0.0), (1.0, 0.0),
                (1.0, 0.0), (0.0, 0.0),
            ],
            dim: 2,
        }
    }

    /// Pauli-Y: maps |0⟩→i|1⟩, |1⟩→−i|0⟩.
    #[must_use]
    pub fn pauli_y() -> Self {
        Self {
            elements: vec![
                (0.0, 0.0), (0.0, -1.0),
                (0.0, 1.0), (0.0, 0.0),
            ],
            dim: 2,
        }
    }

    /// Pauli-Z: phase flip |1⟩→−|1⟩.
    #[must_use]
    pub fn pauli_z() -> Self {
        Self {
            elements: vec![
                (1.0, 0.0), (0.0, 0.0),
                (0.0, 0.0), (-1.0, 0.0),
            ],
            dim: 2,
        }
    }

    /// Hadamard gate: creates superposition.
    #[must_use]
    pub fn hadamard() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            elements: vec![
                (s, 0.0), (s, 0.0),
                (s, 0.0), (-s, 0.0),
            ],
            dim: 2,
        }
    }

    /// Phase gate S: |1⟩→i|1⟩.
    #[must_use]
    pub fn phase_s() -> Self {
        Self {
            elements: vec![
                (1.0, 0.0), (0.0, 0.0),
                (0.0, 0.0), (0.0, 1.0),
            ],
            dim: 2,
        }
    }

    /// T gate (π/8 gate): |1⟩→e^(iπ/4)|1⟩.
    #[must_use]
    pub fn phase_t() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            elements: vec![
                (1.0, 0.0), (0.0, 0.0),
                (0.0, 0.0), (s, s),
            ],
            dim: 2,
        }
    }

    /// Dimension of this operator.
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

    /// Apply this operator to a state vector: |ψ'⟩ = U|ψ⟩.
    pub fn apply(&self, state: &StateVector) -> Result<StateVector> {
        if self.dim != state.dimension() {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: state.dimension(),
            });
        }
        let mut result = vec![(0.0, 0.0); self.dim];
        for i in 0..self.dim {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..self.dim {
                let (m_re, m_im) = self.elements[i * self.dim + j];
                if let Some((s_re, s_im)) = state.amplitude(j) {
                    re += m_re * s_re - m_im * s_im;
                    im += m_re * s_im + m_im * s_re;
                }
            }
            result[i] = (re, im);
        }
        StateVector::new(result).map_err(|_| KanaError::NotUnitary { deviation: 0.0 })
    }

    /// Compute the conjugate transpose (dagger) U†.
    #[must_use]
    pub fn dagger(&self) -> Self {
        let mut elements = vec![(0.0, 0.0); self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                let (re, im) = self.elements[i * self.dim + j];
                elements[j * self.dim + i] = (re, -im);
            }
        }
        Self {
            elements,
            dim: self.dim,
        }
    }

    /// Multiply two operators: C = A × B.
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.dim != other.dim {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        let dim = self.dim;
        let mut elements = vec![(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..dim {
                    let (a_re, a_im) = self.elements[i * dim + k];
                    let (b_re, b_im) = other.elements[k * dim + j];
                    re += a_re * b_re - a_im * b_im;
                    im += a_re * b_im + a_im * b_re;
                }
                elements[i * dim + j] = (re, im);
            }
        }
        Ok(Self { elements, dim })
    }

    /// Tensor product of two operators: A ⊗ B.
    #[must_use]
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_dim = self.dim * other.dim;
        let mut elements = vec![(0.0, 0.0); new_dim * new_dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                let (a_re, a_im) = self.elements[i * self.dim + j];
                for k in 0..other.dim {
                    for l in 0..other.dim {
                        let (b_re, b_im) = other.elements[k * other.dim + l];
                        let row = i * other.dim + k;
                        let col = j * other.dim + l;
                        elements[row * new_dim + col] = (
                            a_re * b_re - a_im * b_im,
                            a_re * b_im + a_im * b_re,
                        );
                    }
                }
            }
        }
        Self {
            elements,
            dim: new_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Operator::identity(2);
        let state = StateVector::zero(1);
        let result = id.apply(&state).unwrap();
        assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_x_flip() {
        let x = Operator::pauli_x();
        let zero = StateVector::zero(1);
        let result = x.apply(&zero).unwrap();
        // |0⟩ → |1⟩
        assert!(result.probability(0).unwrap().abs() < 1e-10);
        assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_x_involution() {
        let x = Operator::pauli_x();
        let xx = x.multiply(&x).unwrap();
        let state = StateVector::plus();
        let result = xx.apply(&state).unwrap();
        // X² = I, so state should be unchanged
        let (re, im) = state.inner_product(&result).unwrap();
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let h = Operator::hadamard();
        let zero = StateVector::zero(1);
        let result = h.apply(&zero).unwrap();
        let p0 = result.probability(0).unwrap();
        let p1 = result.probability(1).unwrap();
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_involution() {
        let h = Operator::hadamard();
        let hh = h.multiply(&h).unwrap();
        let state = StateVector::zero(1);
        let result = hh.apply(&state).unwrap();
        assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_z_phase() {
        let z = Operator::pauli_z();
        let one = StateVector::one();
        let result = z.apply(&one).unwrap();
        // Z|1⟩ = -|1⟩, probability unchanged
        assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dagger() {
        let h = Operator::hadamard();
        let hd = h.dagger();
        // Hadamard is self-adjoint
        for i in 0..2 {
            for j in 0..2 {
                let (a_re, a_im) = h.element(i, j).unwrap();
                let (b_re, b_im) = hd.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_tensor_product() {
        let x = Operator::pauli_x();
        let id = Operator::identity(2);
        let xi = x.tensor_product(&id);
        assert_eq!(xi.dim(), 4);
    }

    #[test]
    fn test_dimension_mismatch() {
        let op = Operator::identity(2);
        let state = StateVector::zero(2); // 4-dim
        assert!(op.apply(&state).is_err());
    }

    #[test]
    fn test_phase_s() {
        let s = Operator::phase_s();
        assert_eq!(s.dim(), 2);
        let (re, im) = s.element(1, 1).unwrap();
        assert!(re.abs() < 1e-10);
        assert!((im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_t() {
        let t = Operator::phase_t();
        assert_eq!(t.dim(), 2);
    }
}
