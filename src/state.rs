//! Quantum state vectors — kets, bras, Hilbert spaces, superposition.
//!
//! A quantum state |ψ⟩ is a unit vector in a complex Hilbert space.
//! For n qubits, the state lives in ℂ^(2^n).

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};

/// Tolerance for floating-point comparisons in normalization and unitarity checks.
pub const NORM_TOLERANCE: f64 = 1e-10;

/// Maximum number of qubits supported (prevents overflow in 2^n dimension).
pub const MAX_QUBITS: usize = 24;

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
        if (norm - 1.0).abs() > NORM_TOLERANCE {
            return Err(KanaError::NotNormalized { norm });
        }
        Ok(state)
    }

    /// Mutable access to amplitudes for in-place gate application.
    #[inline]
    pub(crate) fn amplitudes_mut(&mut self) -> &mut [(f64, f64)] {
        &mut self.amplitudes
    }

    /// Create the |0⟩ state for n qubits (all zeros computational basis state).
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits` is 0 or exceeds `MAX_QUBITS`.
    #[must_use]
    pub fn zero(num_qubits: usize) -> Self {
        assert!(
            num_qubits > 0 && num_qubits <= MAX_QUBITS,
            "num_qubits must be 1..={MAX_QUBITS}, got {num_qubits}"
        );
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

    /// Direct slice access to all amplitudes.
    #[inline]
    #[must_use]
    pub fn amplitudes(&self) -> &[(f64, f64)] {
        &self.amplitudes
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
        self.amplitudes.get(i).map(|(re, im)| re * re + im * im)
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
        for ((a_re, a_im), (b_re, b_im)) in self.amplitudes.iter().zip(other.amplitudes.iter()) {
            // ⟨a|b⟩ = conj(a) * b = (a_re - i*a_im)(b_re + i*b_im)
            re += a_re * b_re + a_im * b_im;
            im += a_re * b_im - a_im * b_re;
        }
        Ok((re, im))
    }

    /// Tensor product |self⟩ ⊗ |other⟩.
    pub fn tensor_product(&self, other: &Self) -> Result<Self> {
        let dim = self
            .amplitudes
            .len()
            .checked_mul(other.amplitudes.len())
            .ok_or_else(|| KanaError::InvalidParameter {
                reason: "tensor product dimension overflow".into(),
            })?;
        let mut amps = Vec::with_capacity(dim);
        for (a_re, a_im) in &self.amplitudes {
            for (b_re, b_im) in &other.amplitudes {
                amps.push((a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re));
            }
        }
        let num_qubits = self.num_qubits + other.num_qubits;
        Ok(Self {
            amplitudes: amps,
            num_qubits,
        })
    }

    /// Measure the full state in the computational basis using Born rule.
    ///
    /// Takes a random value `r` in \[0, 1) to select the outcome deterministically.
    /// Returns `(outcome_index, collapsed_state)`.
    ///
    /// The state collapses to the measured basis state.
    pub fn measure(&self, r: f64) -> Result<(usize, Self)> {
        if !(0.0..1.0).contains(&r) {
            return Err(KanaError::InvalidParameter {
                reason: format!("random value {r} not in [0, 1)"),
            });
        }
        let mut cumulative = 0.0;
        let mut outcome = self.amplitudes.len() - 1;
        for (i, (re, im)) in self.amplitudes.iter().enumerate() {
            cumulative += re * re + im * im;
            if r < cumulative {
                outcome = i;
                break;
            }
        }
        // Collapse to |outcome⟩
        let mut collapsed = vec![(0.0, 0.0); self.amplitudes.len()];
        collapsed[outcome] = (1.0, 0.0);
        Ok((
            outcome,
            Self {
                amplitudes: collapsed,
                num_qubits: self.num_qubits,
            },
        ))
    }

    /// Measure a single qubit, collapsing only that qubit.
    ///
    /// Returns `(bit_value, collapsed_state)` where bit_value is 0 or 1.
    /// The remaining qubits are renormalized but not collapsed.
    pub fn measure_qubit(&self, qubit: usize, r: f64) -> Result<(usize, Self)> {
        if qubit >= self.num_qubits {
            return Err(KanaError::InvalidQubitIndex {
                index: qubit,
                num_qubits: self.num_qubits,
            });
        }
        if !(0.0..1.0).contains(&r) {
            return Err(KanaError::InvalidParameter {
                reason: format!("random value {r} not in [0, 1)"),
            });
        }

        let bit_mask = 1 << (self.num_qubits - 1 - qubit);

        // Calculate probability of measuring |0⟩ on this qubit
        let mut prob_zero = 0.0;
        for (i, (re, im)) in self.amplitudes.iter().enumerate() {
            if i & bit_mask == 0 {
                prob_zero += re * re + im * im;
            }
        }

        let bit_value = if r < prob_zero { 0 } else { 1 };
        let prob_selected = if bit_value == 0 {
            prob_zero
        } else {
            1.0 - prob_zero
        };

        if prob_selected < NORM_TOLERANCE {
            return Err(KanaError::DivisionByZero {
                context: format!("qubit {qubit} measurement probability is zero"),
            });
        }

        // Collapse: zero out amplitudes inconsistent with measurement,
        // renormalize the rest
        let norm_factor = 1.0 / prob_selected.sqrt();
        let collapsed: Vec<(f64, f64)> = self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(i, &(re, im))| {
                let qubit_is_one = (i & bit_mask) != 0;
                if (bit_value == 0 && !qubit_is_one) || (bit_value == 1 && qubit_is_one) {
                    (re * norm_factor, im * norm_factor)
                } else {
                    (0.0, 0.0)
                }
            })
            .collect();

        Ok((
            bit_value,
            Self {
                amplitudes: collapsed,
                num_qubits: self.num_qubits,
            },
        ))
    }

    /// Sample multiple measurement outcomes via Born rule.
    ///
    /// Each `r` value in `random_values` must be in \[0, 1).
    /// Returns a vector of outcome indices (does NOT collapse — each sample
    /// is independent from the original state).
    pub fn sample(&self, random_values: &[f64]) -> Result<Vec<usize>> {
        for &r in random_values {
            if !(0.0..1.0).contains(&r) {
                return Err(KanaError::InvalidParameter {
                    reason: format!("random value {r} not in [0, 1)"),
                });
            }
        }
        let probs = self.probabilities();
        Ok(random_values
            .iter()
            .map(|&r| {
                let mut cumulative = 0.0;
                let mut outcome = probs.len() - 1;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if r < cumulative {
                        outcome = i;
                        break;
                    }
                }
                outcome
            })
            .collect())
    }

    /// Bloch sphere representation for a single-qubit state.
    ///
    /// Returns `(theta, phi)` where the state is:
    /// |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
    ///
    /// Bloch vector: (sin θ cos φ, sin θ sin φ, cos θ)
    /// - θ ∈ \[0, π\]: polar angle (0 = |0⟩, π = |1⟩)
    /// - φ ∈ \[0, 2π): azimuthal angle
    pub fn bloch_angles(&self) -> Result<(f64, f64)> {
        if self.num_qubits != 1 {
            return Err(KanaError::InvalidParameter {
                reason: format!(
                    "Bloch sphere only for single-qubit states, got {} qubits",
                    self.num_qubits
                ),
            });
        }
        let (a_re, a_im) = self.amplitudes[0];
        let (b_re, b_im) = self.amplitudes[1];

        // |α|² = probability of |0⟩ = cos²(θ/2)
        let alpha_abs = (a_re * a_re + a_im * a_im).sqrt();
        let theta = 2.0 * alpha_abs.acos();

        // φ = arg(β) - arg(α)
        let phi = if theta.abs() < NORM_TOLERANCE
            || (std::f64::consts::PI - theta).abs() < NORM_TOLERANCE
        {
            0.0 // at poles, φ is undefined — conventionally 0
        } else {
            let arg_alpha = a_im.atan2(a_re);
            let arg_beta = b_im.atan2(b_re);
            let mut p = arg_beta - arg_alpha;
            if p < 0.0 {
                p += 2.0 * std::f64::consts::PI;
            }
            p
        };

        Ok((theta, phi))
    }

    /// Bloch vector (x, y, z) for a single-qubit state.
    ///
    /// x = sin θ cos φ, y = sin θ sin φ, z = cos θ
    pub fn bloch_vector(&self) -> Result<(f64, f64, f64)> {
        let (theta, phi) = self.bloch_angles()?;
        Ok((
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ))
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
        let zz = z.tensor_product(&o).unwrap();
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

    #[test]
    fn test_measure_deterministic_zero() {
        let s = StateVector::zero(1);
        let (outcome, collapsed) = s.measure(0.5).unwrap();
        assert_eq!(outcome, 0);
        assert!((collapsed.probability(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_deterministic_one() {
        let s = StateVector::one();
        let (outcome, collapsed) = s.measure(0.5).unwrap();
        assert_eq!(outcome, 1);
        assert!((collapsed.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_superposition() {
        let s = StateVector::plus();
        // r < 0.5 → outcome 0
        let (outcome_0, _) = s.measure(0.3).unwrap();
        assert_eq!(outcome_0, 0);
        // r >= 0.5 → outcome 1
        let (outcome_1, _) = s.measure(0.7).unwrap();
        assert_eq!(outcome_1, 1);
    }

    #[test]
    fn test_measure_invalid_r() {
        let s = StateVector::zero(1);
        assert!(s.measure(1.0).is_err());
        assert!(s.measure(-0.1).is_err());
    }

    #[test]
    fn test_measure_qubit_bell() {
        // |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let bell = StateVector::new(vec![(s, 0.0), (0.0, 0.0), (0.0, 0.0), (s, 0.0)]).unwrap();

        // Measure qubit 0, r=0.3 → should get 0, collapse to |00⟩
        let (bit, collapsed) = bell.measure_qubit(0, 0.3).unwrap();
        assert_eq!(bit, 0);
        assert!((collapsed.probability(0).unwrap() - 1.0).abs() < 1e-10);

        // Measure qubit 0, r=0.7 → should get 1, collapse to |11⟩
        let (bit, collapsed) = bell.measure_qubit(0, 0.7).unwrap();
        assert_eq!(bit, 1);
        assert!((collapsed.probability(3).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_qubit_oob() {
        let s = StateVector::zero(1);
        assert!(s.measure_qubit(5, 0.5).is_err());
    }

    #[test]
    fn test_sample_deterministic() {
        let s = StateVector::zero(1);
        let outcomes = s.sample(&[0.0, 0.5, 0.99]).unwrap();
        assert_eq!(outcomes, vec![0, 0, 0]);
    }

    #[test]
    fn test_sample_superposition() {
        let s = StateVector::plus();
        let outcomes = s.sample(&[0.1, 0.3, 0.6, 0.9]).unwrap();
        // r < 0.5 → 0, r >= 0.5 → 1
        assert_eq!(outcomes, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_bloch_zero_state() {
        let s = StateVector::zero(1);
        let (theta, _phi) = s.bloch_angles().unwrap();
        assert!(theta.abs() < 1e-10); // north pole
        let (x, y, z) = s.bloch_vector().unwrap();
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!((z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bloch_one_state() {
        let s = StateVector::one();
        let (theta, _phi) = s.bloch_angles().unwrap();
        assert!((theta - std::f64::consts::PI).abs() < 1e-10); // south pole
        let (x, y, z) = s.bloch_vector().unwrap();
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!((z - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_bloch_plus_state() {
        let s = StateVector::plus();
        let (x, y, z) = s.bloch_vector().unwrap();
        // |+⟩ is on the +x axis
        assert!((x - 1.0).abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_bloch_minus_state() {
        let s = StateVector::minus();
        let (x, y, z) = s.bloch_vector().unwrap();
        // |−⟩ is on the −x axis
        assert!((x - (-1.0)).abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_bloch_multi_qubit_rejected() {
        let s = StateVector::zero(2);
        assert!(s.bloch_angles().is_err());
    }
}
