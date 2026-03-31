//! Quantum state vectors — kets, bras, Hilbert spaces, superposition.
//!
//! A quantum state |ψ⟩ is a unit vector in a complex Hilbert space.
//! For n qubits, the state lives in ℂ^(2^n).

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};

/// A quantum state vector in the computational basis.
///
/// Stores complex amplitudes as pairs of (real, imaginary) components.
/// The state must be normalized: Σ|αᵢ|² = 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVector {
    /// Complex amplitudes as (re, im) pairs.
    amplitudes: Vec<(f64, f64)>,
    /// Number of qubits in the system.
    num_qubits: usize,
}

impl StateVector {
    /// Create a new state vector from complex amplitudes.
    ///
    /// Validates that the dimension is a power of 2 and the state is normalized.
    pub fn new(amplitudes: Vec<(f64, f64)>) -> Result<Self> {
        let dim = amplitudes.len();
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return Err(KanaError::InvalidParameter {
                reason: format!("dimension {dim} is not a power of 2"),
            });
        }
        let num_qubits = dim.trailing_zeros() as usize;
        let state = Self {
            amplitudes,
            num_qubits,
        };
        let norm = state.norm();
        if (norm - 1.0).abs() > 1e-10 {
            return Err(KanaError::NotNormalized { norm });
        }
        Ok(state)
    }

    /// Create the |0⟩ state for n qubits (all zeros computational basis state).
    #[must_use]
    pub fn zero(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amps = vec![(0.0, 0.0); dim];
        amps[0] = (1.0, 0.0);
        Self {
            amplitudes: amps,
            num_qubits,
        }
    }

    /// Create the |1⟩ state for a single qubit.
    #[must_use]
    pub fn one() -> Self {
        Self {
            amplitudes: vec![(0.0, 0.0), (1.0, 0.0)],
            num_qubits: 1,
        }
    }

    /// Create an equal superposition state: (|0⟩ + |1⟩)/√2 for a single qubit.
    #[must_use]
    pub fn plus() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            amplitudes: vec![(s, 0.0), (s, 0.0)],
            num_qubits: 1,
        }
    }

    /// Create the |−⟩ state: (|0⟩ − |1⟩)/√2.
    #[must_use]
    pub fn minus() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            amplitudes: vec![(s, 0.0), (-s, 0.0)],
            num_qubits: 1,
        }
    }

    /// Number of qubits in this state.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Dimension of the Hilbert space (2^n).
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// Get the complex amplitude at index i.
    #[inline]
    #[must_use]
    pub fn amplitude(&self, i: usize) -> Option<(f64, f64)> {
        self.amplitudes.get(i).copied()
    }

    /// Compute the norm (should be 1.0 for valid states).
    #[inline]
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>()
            .sqrt()
    }

    /// Probability of measuring basis state |i⟩.
    #[inline]
    #[must_use]
    pub fn probability(&self, i: usize) -> Option<f64> {
        self.amplitudes
            .get(i)
            .map(|(re, im)| re * re + im * im)
    }

    /// All probabilities for each basis state.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|(re, im)| re * re + im * im)
            .collect()
    }

    /// Inner product ⟨self|other⟩.
    pub fn inner_product(&self, other: &Self) -> Result<(f64, f64)> {
        if self.amplitudes.len() != other.amplitudes.len() {
            return Err(KanaError::DimensionMismatch {
                expected: self.amplitudes.len(),
                got: other.amplitudes.len(),
            });
        }
        let (mut re, mut im) = (0.0, 0.0);
        for ((a_re, a_im), (b_re, b_im)) in
            self.amplitudes.iter().zip(other.amplitudes.iter())
        {
            // ⟨a|b⟩ = conj(a) * b = (a_re - i*a_im)(b_re + i*b_im)
            re += a_re * b_re + a_im * b_im;
            im += a_re * b_im - a_im * b_re;
        }
        Ok((re, im))
    }

    /// Tensor product |self⟩ ⊗ |other⟩.
    #[must_use]
    pub fn tensor_product(&self, other: &Self) -> Self {
        let dim = self.amplitudes.len() * other.amplitudes.len();
        let mut amps = Vec::with_capacity(dim);
        for (a_re, a_im) in &self.amplitudes {
            for (b_re, b_im) in &other.amplitudes {
                amps.push((
                    a_re * b_re - a_im * b_im,
                    a_re * b_im + a_im * b_re,
                ));
            }
        }
        let num_qubits = self.num_qubits + other.num_qubits;
        Self {
            amplitudes: amps,
            num_qubits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let s = StateVector::zero(1);
        assert_eq!(s.num_qubits(), 1);
        assert_eq!(s.dimension(), 2);
        assert_eq!(s.amplitude(0), Some((1.0, 0.0)));
        assert_eq!(s.amplitude(1), Some((0.0, 0.0)));
    }

    #[test]
    fn test_one_state() {
        let s = StateVector::one();
        assert_eq!(s.probability(0), Some(0.0));
        assert!((s.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plus_state() {
        let s = StateVector::plus();
        let p0 = s.probability(0).unwrap();
        let p1 = s.probability(1).unwrap();
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minus_state() {
        let s = StateVector::minus();
        let probs = s.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_norm() {
        let s = StateVector::zero(2);
        assert!((s.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let z = StateVector::zero(1);
        let o = StateVector::one();
        let (re, im) = z.inner_product(&o).unwrap();
        assert!(re.abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_inner_product_self() {
        let s = StateVector::plus();
        let (re, im) = s.inner_product(&s).unwrap();
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_tensor_product() {
        let z = StateVector::zero(1);
        let o = StateVector::one();
        let zz = z.tensor_product(&o);
        assert_eq!(zz.num_qubits(), 2);
        assert_eq!(zz.dimension(), 4);
        // |0⟩⊗|1⟩ = |01⟩ → index 1
        assert!((zz.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_validates_normalization() {
        let result = StateVector::new(vec![(1.0, 0.0), (1.0, 0.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_validates_power_of_two() {
        let result = StateVector::new(vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_valid() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let result = StateVector::new(vec![(s, 0.0), (s, 0.0)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_qubit_zero() {
        let s = StateVector::zero(3);
        assert_eq!(s.dimension(), 8);
        assert!((s.probability(0).unwrap() - 1.0).abs() < 1e-10);
        for i in 1..8 {
            assert!(s.probability(i).unwrap().abs() < 1e-10);
        }
    }

    #[test]
    fn test_dimension_mismatch_inner() {
        let a = StateVector::zero(1);
        let b = StateVector::zero(2);
        assert!(a.inner_product(&b).is_err());
    }
}
