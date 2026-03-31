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
        if dim == 0 {
            return Err(KanaError::InvalidParameter {
                reason: "operator dimension must be > 0".into(),
            });
        }
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
            elements: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
            dim: 2,
        }
    }

    /// Pauli-Y: maps |0⟩→i|1⟩, |1⟩→−i|0⟩.
    #[must_use]
    pub fn pauli_y() -> Self {
        Self {
            elements: vec![(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)],
            dim: 2,
        }
    }

    /// Pauli-Z: phase flip |1⟩→−|1⟩.
    #[must_use]
    pub fn pauli_z() -> Self {
        Self {
            elements: vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)],
            dim: 2,
        }
    }

    /// Hadamard gate: creates superposition.
    #[must_use]
    pub fn hadamard() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            elements: vec![(s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0)],
            dim: 2,
        }
    }

    /// Phase gate S: |1⟩→i|1⟩.
    #[must_use]
    pub fn phase_s() -> Self {
        Self {
            elements: vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)],
            dim: 2,
        }
    }

    /// T gate (π/8 gate): |1⟩→e^(iπ/4)|1⟩.
    #[must_use]
    pub fn phase_t() -> Self {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            elements: vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s, s)],
            dim: 2,
        }
    }

    /// Rotation around X axis: Rx(θ) = exp(−iθX/2).
    ///
    /// Rx(θ) = [[cos(θ/2), −i·sin(θ/2)], [−i·sin(θ/2), cos(θ/2)]]
    #[must_use]
    pub fn rx(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        Self {
            elements: vec![(c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0)],
            dim: 2,
        }
    }

    /// Rotation around Y axis: Ry(θ) = exp(−iθY/2).
    ///
    /// Ry(θ) = [[cos(θ/2), −sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    #[must_use]
    pub fn ry(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        Self {
            elements: vec![(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)],
            dim: 2,
        }
    }

    /// Rotation around Z axis: Rz(θ) = exp(−iθZ/2).
    ///
    /// Rz(θ) = [[e^(−iθ/2), 0], [0, e^(iθ/2)]]
    #[must_use]
    pub fn rz(theta: f64) -> Self {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        Self {
            elements: vec![(c, -s), (0.0, 0.0), (0.0, 0.0), (c, s)],
            dim: 2,
        }
    }

    /// Phase gate with arbitrary angle: |1⟩ → e^(iφ)|1⟩.
    #[must_use]
    pub fn phase(phi: f64) -> Self {
        Self {
            elements: vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (phi.cos(), phi.sin())],
            dim: 2,
        }
    }

    /// CNOT (controlled-X) gate on 2 qubits.
    ///
    /// |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    /// Control is qubit 0, target is qubit 1.
    #[must_use]
    pub fn cnot() -> Self {
        Self {
            elements: vec![
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
            ],
            dim: 4,
        }
    }

    /// Controlled-Z gate on 2 qubits.
    ///
    /// |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|10⟩, |11⟩→−|11⟩
    #[must_use]
    pub fn cz() -> Self {
        Self {
            elements: vec![
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (-1.0, 0.0),
            ],
            dim: 4,
        }
    }

    /// SWAP gate on 2 qubits.
    ///
    /// |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    #[must_use]
    pub fn swap() -> Self {
        Self {
            elements: vec![
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
            ],
            dim: 4,
        }
    }

    /// Toffoli gate (CCX / CCNOT) on 3 qubits.
    ///
    /// Flips qubit 2 iff qubits 0 and 1 are both |1⟩.
    /// |110⟩→|111⟩, |111⟩→|110⟩, all others unchanged.
    #[must_use]
    pub fn toffoli() -> Self {
        let mut elements = vec![(0.0, 0.0); 64];
        // Identity on all basis states except |110⟩↔|111⟩
        for i in 0..8 {
            elements[i * 8 + i] = (1.0, 0.0);
        }
        // Swap |110⟩ (6) and |111⟩ (7)
        elements[6 * 8 + 6] = (0.0, 0.0);
        elements[7 * 8 + 7] = (0.0, 0.0);
        elements[6 * 8 + 7] = (1.0, 0.0);
        elements[7 * 8 + 6] = (1.0, 0.0);
        Self { elements, dim: 8 }
    }

    /// Fredkin gate (CSWAP) on 3 qubits.
    ///
    /// Swaps qubits 1 and 2 iff qubit 0 is |1⟩.
    /// |101⟩↔|110⟩, all others unchanged.
    #[must_use]
    pub fn fredkin() -> Self {
        let mut elements = vec![(0.0, 0.0); 64];
        for i in 0..8 {
            elements[i * 8 + i] = (1.0, 0.0);
        }
        // Swap |101⟩ (5) and |110⟩ (6)
        elements[5 * 8 + 5] = (0.0, 0.0);
        elements[6 * 8 + 6] = (0.0, 0.0);
        elements[5 * 8 + 6] = (1.0, 0.0);
        elements[6 * 8 + 5] = (1.0, 0.0);
        Self { elements, dim: 8 }
    }

    /// Build a controlled-U gate from a single-qubit operator.
    ///
    /// Returns a 4×4 operator: applies U to the target qubit when
    /// the control qubit is |1⟩. Control is qubit 0, target is qubit 1.
    pub fn controlled(u: &Self) -> Result<Self> {
        if u.dim != 2 {
            return Err(KanaError::DimensionMismatch {
                expected: 2,
                got: u.dim,
            });
        }
        // |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
        let mut elements = vec![(0.0, 0.0); 16];
        // Top-left 2×2: identity (control=0)
        elements[0] = (1.0, 0.0); // (0,0)
        elements[1] = (0.0, 0.0); // (0,1)
        elements[4] = (0.0, 0.0); // (1,0)
        elements[5] = (1.0, 0.0); // (1,1)
        // Bottom-right 2×2: U (control=1)
        elements[10] = u.elements[0]; // (2,2) = U[0,0]
        elements[11] = u.elements[1]; // (2,3) = U[0,1]
        elements[14] = u.elements[2]; // (3,2) = U[1,0]
        elements[15] = u.elements[3]; // (3,3) = U[1,1]
        Ok(Self { elements, dim: 4 })
    }

    /// Dimension of this operator.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Direct slice access to all matrix elements (row-major).
    #[inline]
    #[must_use]
    pub fn elements(&self) -> &[(f64, f64)] {
        &self.elements
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
    #[inline]
    pub fn apply(&self, state: &StateVector) -> Result<StateVector> {
        if self.dim != state.dimension() {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: state.dimension(),
            });
        }
        let amps = state.amplitudes();
        let mut result = vec![(0.0, 0.0); self.dim];
        for (i, slot) in result.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0, 0.0);
            let row_start = i * self.dim;
            for (j, &(s_re, s_im)) in amps.iter().enumerate() {
                let (m_re, m_im) = self.elements[row_start + j];
                re += m_re * s_re - m_im * s_im;
                im += m_re * s_im + m_im * s_re;
            }
            *slot = (re, im);
        }
        StateVector::new(result).map_err(|e| match e {
            KanaError::NotNormalized { norm } => KanaError::NotUnitary {
                deviation: (norm - 1.0).abs(),
            },
            other => other,
        })
    }

    /// Compute the conjugate transpose (dagger) U†.
    #[inline]
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
    #[inline]
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
                        elements[row * new_dim + col] =
                            (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
                    }
                }
            }
        }
        Self {
            elements,
            dim: new_dim,
        }
    }

    /// Convert to sparse representation, dropping entries below tolerance.
    #[must_use]
    pub fn to_sparse(&self) -> SparseOperator {
        SparseOperator::from_dense(self)
    }

    /// Sparsity ratio: fraction of zero elements.
    #[inline]
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let total = self.elements.len();
        let zeros = self
            .elements
            .iter()
            .filter(|&&(re, im)| re == 0.0 && im == 0.0)
            .count();
        zeros as f64 / total as f64
    }
}

/// Sparse quantum operator in COO (coordinate) format.
///
/// Stores only non-zero (row, col, value) entries. Efficient for
/// gates with many zeros (CNOT, Toffoli, identity-like operators).
#[derive(Debug, Clone)]
pub struct SparseOperator {
    /// Non-zero entries as (row, col, re, im).
    entries: Vec<(usize, usize, f64, f64)>,
    /// Dimension of the operator (n×n).
    dim: usize,
}

impl SparseOperator {
    /// Create a sparse operator from a list of non-zero entries.
    pub fn new(dim: usize, entries: Vec<(usize, usize, f64, f64)>) -> Result<Self> {
        if dim == 0 {
            return Err(KanaError::InvalidParameter {
                reason: "operator dimension must be > 0".into(),
            });
        }
        for &(row, col, _, _) in &entries {
            if row >= dim || col >= dim {
                return Err(KanaError::InvalidParameter {
                    reason: format!("entry ({row}, {col}) out of bounds for dim {dim}"),
                });
            }
        }
        Ok(Self { entries, dim })
    }

    /// Convert from a dense operator, dropping entries with magnitude below tolerance.
    #[must_use]
    pub fn from_dense(op: &Operator) -> Self {
        let dim = op.dim();
        let tol = crate::state::NORM_TOLERANCE;
        let entries: Vec<(usize, usize, f64, f64)> = op
            .elements()
            .iter()
            .enumerate()
            .filter_map(|(idx, &(re, im))| {
                if re.abs() > tol || im.abs() > tol {
                    let row = idx / dim;
                    let col = idx % dim;
                    Some((row, col, re, im))
                } else {
                    None
                }
            })
            .collect();
        Self { entries, dim }
    }

    /// Convert back to a dense operator.
    #[must_use]
    pub fn to_dense(&self) -> Operator {
        let mut elements = vec![(0.0, 0.0); self.dim * self.dim];
        for &(row, col, re, im) in &self.entries {
            elements[row * self.dim + col] = (re, im);
        }
        Operator {
            elements,
            dim: self.dim,
        }
    }

    /// Dimension of this operator.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of non-zero entries.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Apply this sparse operator to a state vector: |ψ'⟩ = U|ψ⟩.
    ///
    /// Only multiplies by non-zero entries, skipping all zeros.
    /// For a gate with k non-zero entries out of n², this is O(k) per
    /// output element instead of O(n).
    pub fn apply(&self, state: &StateVector) -> Result<StateVector> {
        if self.dim != state.dimension() {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: state.dimension(),
            });
        }
        let amps = state.amplitudes();
        let mut result = vec![(0.0, 0.0); self.dim];
        for &(row, col, m_re, m_im) in &self.entries {
            let (s_re, s_im) = amps[col];
            result[row].0 += m_re * s_re - m_im * s_im;
            result[row].1 += m_re * s_im + m_im * s_re;
        }
        StateVector::new(result).map_err(|e| match e {
            KanaError::NotNormalized { norm } => KanaError::NotUnitary {
                deviation: (norm - 1.0).abs(),
            },
            other => other,
        })
    }

    /// Sparse matrix-matrix multiply.
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.dim != other.dim {
            return Err(KanaError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        let dim = self.dim;
        // Build result: for each (i,k) in self and (k,j) in other, add to (i,j)
        let mut dense = vec![(0.0, 0.0); dim * dim];
        for &(i, k, a_re, a_im) in &self.entries {
            for &(k2, j, b_re, b_im) in &other.entries {
                if k == k2 {
                    let idx = i * dim + j;
                    dense[idx].0 += a_re * b_re - a_im * b_im;
                    dense[idx].1 += a_re * b_im + a_im * b_re;
                }
            }
        }
        // Convert back to sparse
        let tol = crate::state::NORM_TOLERANCE;
        let entries: Vec<(usize, usize, f64, f64)> = dense
            .iter()
            .enumerate()
            .filter_map(|(idx, &(re, im))| {
                if re.abs() > tol || im.abs() > tol {
                    Some((idx / dim, idx % dim, re, im))
                } else {
                    None
                }
            })
            .collect();
        Ok(Self { entries, dim })
    }

    /// Sparse Kronecker (tensor) product.
    #[must_use]
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_dim = self.dim * other.dim;
        let mut entries = Vec::with_capacity(self.entries.len() * other.entries.len());
        for &(i, j, a_re, a_im) in &self.entries {
            for &(k, l, b_re, b_im) in &other.entries {
                let row = i * other.dim + k;
                let col = j * other.dim + l;
                let re = a_re * b_re - a_im * b_im;
                let im = a_re * b_im + a_im * b_re;
                entries.push((row, col, re, im));
            }
        }
        Self {
            entries,
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

    #[test]
    fn test_cnot_computational_basis() {
        let cnot = Operator::cnot();
        assert_eq!(cnot.dim(), 4);
        // |00⟩ → |00⟩
        let s00 = StateVector::zero(2);
        let r00 = cnot.apply(&s00).unwrap();
        assert!((r00.probability(0).unwrap() - 1.0).abs() < 1e-10);

        // |10⟩ → |11⟩
        let s10 = StateVector::new(vec![(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]).unwrap();
        let r10 = cnot.apply(&s10).unwrap();
        assert!((r10.probability(3).unwrap() - 1.0).abs() < 1e-10);

        // |11⟩ → |10⟩
        let s11 = StateVector::new(vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]).unwrap();
        let r11 = cnot.apply(&s11).unwrap();
        assert!((r11.probability(2).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_involution() {
        let cnot = Operator::cnot();
        let cnot2 = cnot.multiply(&cnot).unwrap();
        // CNOT² = I
        let id = Operator::identity(4);
        for i in 0..4 {
            for j in 0..4 {
                let (a_re, a_im) = cnot2.element(i, j).unwrap();
                let (b_re, b_im) = id.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cz_gate() {
        let cz = Operator::cz();
        assert_eq!(cz.dim(), 4);
        // |11⟩ → −|11⟩
        let s11 = StateVector::new(vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]).unwrap();
        let r = cz.apply(&s11).unwrap();
        let (re, im) = r.amplitude(3).unwrap();
        assert!((re - (-1.0)).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_swap_gate() {
        let swap = Operator::swap();
        // |01⟩ → |10⟩
        let s01 = StateVector::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]).unwrap();
        let r = swap.apply(&s01).unwrap();
        assert!((r.probability(2).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli() {
        let tof = Operator::toffoli();
        assert_eq!(tof.dim(), 8);
        // |110⟩ (idx 6) → |111⟩ (idx 7)
        let mut amps = vec![(0.0, 0.0); 8];
        amps[6] = (1.0, 0.0);
        let s = StateVector::new(amps).unwrap();
        let r = tof.apply(&s).unwrap();
        assert!((r.probability(7).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli_no_flip() {
        let tof = Operator::toffoli();
        // |100⟩ (idx 4) → |100⟩ (control_b is 0, no flip)
        let mut amps = vec![(0.0, 0.0); 8];
        amps[4] = (1.0, 0.0);
        let s = StateVector::new(amps).unwrap();
        let r = tof.apply(&s).unwrap();
        assert!((r.probability(4).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli_involution() {
        let tof = Operator::toffoli();
        let tof2 = tof.multiply(&tof).unwrap();
        let id = Operator::identity(8);
        for i in 0..8 {
            for j in 0..8 {
                let (a_re, a_im) = tof2.element(i, j).unwrap();
                let (b_re, b_im) = id.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_fredkin() {
        let fred = Operator::fredkin();
        assert_eq!(fred.dim(), 8);
        // |101⟩ (idx 5) → |110⟩ (idx 6)
        let mut amps = vec![(0.0, 0.0); 8];
        amps[5] = (1.0, 0.0);
        let s = StateVector::new(amps).unwrap();
        let r = fred.apply(&s).unwrap();
        assert!((r.probability(6).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fredkin_no_swap() {
        let fred = Operator::fredkin();
        // |010⟩ (idx 2) → |010⟩ (control is 0, no swap)
        let mut amps = vec![(0.0, 0.0); 8];
        amps[2] = (1.0, 0.0);
        let s = StateVector::new(amps).unwrap();
        let r = fred.apply(&s).unwrap();
        assert!((r.probability(2).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_controlled_u() {
        // Controlled-Z = CZ
        let cz_built = Operator::controlled(&Operator::pauli_z()).unwrap();
        let cz = Operator::cz();
        for i in 0..4 {
            for j in 0..4 {
                let (a_re, a_im) = cz_built.element(i, j).unwrap();
                let (b_re, b_im) = cz.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_controlled_u_dimension_check() {
        let id4 = Operator::identity(4);
        assert!(Operator::controlled(&id4).is_err());
    }

    #[test]
    fn test_sparse_roundtrip() {
        let h = Operator::hadamard();
        let sparse = h.to_sparse();
        let back = sparse.to_dense();
        for i in 0..2 {
            for j in 0..2 {
                let (a_re, a_im) = h.element(i, j).unwrap();
                let (b_re, b_im) = back.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_nnz() {
        // Identity 4x4 has 4 non-zero entries
        let id = Operator::identity(4);
        let sparse = id.to_sparse();
        assert_eq!(sparse.nnz(), 4);
        assert_eq!(sparse.dim(), 4);

        // CNOT has 4 non-zero entries
        let cnot = Operator::cnot();
        assert_eq!(cnot.to_sparse().nnz(), 4);
    }

    #[test]
    fn test_sparse_apply_matches_dense() {
        let h = Operator::hadamard();
        let sparse = h.to_sparse();
        let state = StateVector::zero(1);

        let dense_result = h.apply(&state).unwrap();
        let sparse_result = sparse.apply(&state).unwrap();

        for i in 0..2 {
            let (a_re, a_im) = dense_result.amplitude(i).unwrap();
            let (b_re, b_im) = sparse_result.amplitude(i).unwrap();
            assert!((a_re - b_re).abs() < 1e-10);
            assert!((a_im - b_im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_tensor_product() {
        let x = Operator::pauli_x().to_sparse();
        let id = Operator::identity(2).to_sparse();
        let xi = x.tensor_product(&id);
        assert_eq!(xi.dim(), 4);
        assert_eq!(xi.nnz(), 4); // X⊗I has 4 non-zero entries
    }

    #[test]
    fn test_sparsity() {
        let id = Operator::identity(4);
        assert!((id.sparsity() - 0.75).abs() < 1e-10); // 12/16 zeros

        let tof = Operator::toffoli();
        assert!((tof.sparsity() - 0.875).abs() < 1e-10); // 56/64 zeros
    }
}
