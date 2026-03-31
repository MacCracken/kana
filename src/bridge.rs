//! Hisab bridge — convert between kana types and hisab linear algebra types.
//!
//! This module provides conversion functions that map kana's quantum types
//! (`StateVector`, `Operator`, `DensityMatrix`) to and from hisab's
//! `Complex`, `ComplexMatrix`, and related types.
//!
//! # Architecture
//!
//! ```text
//! kana (quantum)  ←→  bridge  ←→  hisab (linear algebra)
//!   StateVector    ←→            ←→  Vec<Complex>
//!   Operator       ←→            ←→  ComplexMatrix
//!   DensityMatrix  ←→            ←→  ComplexMatrix
//! ```
//!
//! Always feature-gated behind `hisab-bridge`. Takes kana types, returns
//! hisab types (and vice versa). No wrapper types — pure conversion functions.

use hisab::num::Complex;
use hisab::num::ComplexMatrix;

use crate::error::{KanaError, Result};

// ---------------------------------------------------------------------------
// Complex number conversions
// ---------------------------------------------------------------------------

/// Convert a kana complex tuple `(re, im)` to a hisab `Complex`.
#[inline]
#[must_use]
pub fn to_complex(c: (f64, f64)) -> Complex {
    Complex::new(c.0, c.1)
}

/// Convert a hisab `Complex` to a kana complex tuple `(re, im)`.
#[inline]
#[must_use]
pub fn from_complex(c: Complex) -> (f64, f64) {
    (c.re, c.im)
}

/// Convert a slice of kana complex tuples to a `Vec<Complex>`.
#[must_use]
pub fn to_complex_vec(tuples: &[(f64, f64)]) -> Vec<Complex> {
    tuples.iter().map(|&c| to_complex(c)).collect()
}

/// Convert a slice of hisab `Complex` values to kana tuples.
#[must_use]
pub fn from_complex_vec(complexes: &[Complex]) -> Vec<(f64, f64)> {
    complexes.iter().map(|&c| from_complex(c)).collect()
}

// ---------------------------------------------------------------------------
// StateVector ↔ Vec<Complex>
// ---------------------------------------------------------------------------

/// Extract amplitudes from a `StateVector` as hisab `Complex` values.
#[cfg(feature = "state")]
#[must_use]
pub fn state_to_complex(state: &crate::state::StateVector) -> Vec<Complex> {
    to_complex_vec(state.amplitudes())
}

/// Create a `StateVector` from hisab `Complex` amplitudes.
#[cfg(feature = "state")]
pub fn state_from_complex(amplitudes: &[Complex]) -> Result<crate::state::StateVector> {
    crate::state::StateVector::new(from_complex_vec(amplitudes))
}

// ---------------------------------------------------------------------------
// Operator ↔ ComplexMatrix
// ---------------------------------------------------------------------------

/// Convert a kana `Operator` to a hisab `ComplexMatrix`.
#[cfg(feature = "operator")]
pub fn operator_to_matrix(op: &crate::operator::Operator) -> Result<ComplexMatrix> {
    let dim = op.dim();
    let data = to_complex_vec(op.elements());
    ComplexMatrix::from_rows(dim, dim, data).map_err(|e| KanaError::InvalidParameter {
        reason: format!("failed to convert operator to ComplexMatrix: {e}"),
    })
}

/// Create a kana `Operator` from a hisab `ComplexMatrix`.
#[cfg(feature = "operator")]
pub fn operator_from_matrix(m: &ComplexMatrix) -> Result<crate::operator::Operator> {
    if m.rows() != m.cols() {
        return Err(KanaError::DimensionMismatch {
            expected: m.rows(),
            got: m.cols(),
        });
    }
    let dim = m.rows();
    let mut elements = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            elements.push(from_complex(m.get(i, j)));
        }
    }
    crate::operator::Operator::new(dim, elements)
}

// ---------------------------------------------------------------------------
// DensityMatrix ↔ ComplexMatrix
// ---------------------------------------------------------------------------

/// Convert a kana `DensityMatrix` to a hisab `ComplexMatrix`.
#[cfg(feature = "entanglement")]
pub fn density_to_matrix(dm: &crate::entanglement::DensityMatrix) -> Result<ComplexMatrix> {
    let dim = dm.dim();
    let mut data = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            let (re, im) = dm.element(i, j).ok_or(KanaError::InvalidParameter {
                reason: "density matrix element out of bounds".into(),
            })?;
            data.push(Complex::new(re, im));
        }
    }
    ComplexMatrix::from_rows(dim, dim, data).map_err(|e| KanaError::InvalidParameter {
        reason: format!("failed to convert density matrix: {e}"),
    })
}

/// Create a kana `DensityMatrix` from a hisab `ComplexMatrix`.
#[cfg(feature = "entanglement")]
pub fn density_from_matrix(m: &ComplexMatrix) -> Result<crate::entanglement::DensityMatrix> {
    if m.rows() != m.cols() {
        return Err(KanaError::DimensionMismatch {
            expected: m.rows(),
            got: m.cols(),
        });
    }
    let dim = m.rows();
    let mut elements = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            elements.push(from_complex(m.get(i, j)));
        }
    }
    crate::entanglement::DensityMatrix::new(dim, elements)
}

// ---------------------------------------------------------------------------
// Hisab utilities bridged into kana domain
// ---------------------------------------------------------------------------

/// Check if an operator is unitary using hisab's `ComplexMatrix::is_unitary`.
#[cfg(feature = "operator")]
pub fn is_unitary(op: &crate::operator::Operator, tol: f64) -> Result<bool> {
    let m = operator_to_matrix(op)?;
    Ok(m.is_unitary(tol))
}

/// Check if a density matrix is Hermitian using hisab's `ComplexMatrix::is_hermitian`.
#[cfg(feature = "entanglement")]
pub fn is_hermitian(dm: &crate::entanglement::DensityMatrix, tol: f64) -> Result<bool> {
    let m = density_to_matrix(dm)?;
    Ok(m.is_hermitian(tol))
}

/// Compute the Kronecker (tensor) product of two operators using hisab.
#[cfg(feature = "operator")]
pub fn kronecker(
    a: &crate::operator::Operator,
    b: &crate::operator::Operator,
) -> Result<crate::operator::Operator> {
    let ma = operator_to_matrix(a)?;
    let mb = operator_to_matrix(b)?;
    let result = hisab::num::kronecker(&ma, &mb);
    operator_from_matrix(&result)
}

/// Compute eigenvalues of a Hermitian density matrix using hisab's
/// `eigen_hermitian` (QR-based, more robust than kana's built-in Jacobi).
#[cfg(feature = "entanglement")]
pub fn eigenvalues_hermitian(dm: &crate::entanglement::DensityMatrix) -> Result<Vec<f64>> {
    let m = density_to_matrix(dm)?;
    let decomp =
        hisab::num::eigen_hermitian(&m, 1e-12, 1000).map_err(|e| KanaError::InvalidParameter {
            reason: format!("eigendecomposition failed: {e}"),
        })?;
    Ok(decomp.eigenvalues)
}

/// Compute the matrix exponential of an operator using hisab.
///
/// Useful for time evolution: U(t) = exp(−iHt).
#[cfg(feature = "operator")]
pub fn matrix_exp(op: &crate::operator::Operator) -> Result<crate::operator::Operator> {
    let m = operator_to_matrix(op)?;
    let result = hisab::num::matrix_exp(&m).map_err(|e| KanaError::InvalidParameter {
        reason: format!("matrix exponential failed: {e}"),
    })?;
    operator_from_matrix(&result)
}

/// Compute the commutator [A, B] = AB − BA using hisab.
#[cfg(feature = "operator")]
pub fn commutator(
    a: &crate::operator::Operator,
    b: &crate::operator::Operator,
) -> Result<crate::operator::Operator> {
    let ma = operator_to_matrix(a)?;
    let mb = operator_to_matrix(b)?;
    let result = hisab::num::commutator(&ma, &mb).map_err(|e| KanaError::InvalidParameter {
        reason: format!("commutator failed: {e}"),
    })?;
    operator_from_matrix(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_roundtrip() {
        let c = (0.5, -0.3);
        assert_eq!(from_complex(to_complex(c)), c);
    }

    #[test]
    fn test_complex_vec_roundtrip() {
        let v = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.5)];
        assert_eq!(from_complex_vec(&to_complex_vec(&v)), v);
    }

    #[cfg(feature = "state")]
    #[test]
    fn test_state_roundtrip() {
        let state = crate::state::StateVector::zero(2);
        let complex = state_to_complex(&state);
        let back = state_from_complex(&complex).unwrap();
        for i in 0..state.dimension() {
            let (a_re, a_im) = state.amplitude(i).unwrap();
            let (b_re, b_im) = back.amplitude(i).unwrap();
            assert!((a_re - b_re).abs() < 1e-15);
            assert!((a_im - b_im).abs() < 1e-15);
        }
    }

    #[cfg(feature = "operator")]
    #[test]
    fn test_operator_roundtrip() {
        let op = crate::operator::Operator::hadamard();
        let matrix = operator_to_matrix(&op).unwrap();
        let back = operator_from_matrix(&matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let (a_re, a_im) = op.element(i, j).unwrap();
                let (b_re, b_im) = back.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-15);
                assert!((a_im - b_im).abs() < 1e-15);
            }
        }
    }

    #[cfg(feature = "operator")]
    #[test]
    fn test_is_unitary_hadamard() {
        let h = crate::operator::Operator::hadamard();
        assert!(is_unitary(&h, 1e-10).unwrap());
    }

    #[cfg(feature = "operator")]
    #[test]
    fn test_kronecker_matches_kana() {
        let x = crate::operator::Operator::pauli_x();
        let id = crate::operator::Operator::identity(2);
        let kana_result = x.tensor_product(&id);
        let bridge_result = kronecker(&x, &id).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let (a_re, a_im) = kana_result.element(i, j).unwrap();
                let (b_re, b_im) = bridge_result.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-10);
                assert!((a_im - b_im).abs() < 1e-10);
            }
        }
    }

    #[cfg(feature = "entanglement")]
    #[test]
    fn test_density_roundtrip() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        let dm = crate::entanglement::DensityMatrix::from_pure_state(&amps);
        let matrix = density_to_matrix(&dm).unwrap();
        let back = density_from_matrix(&matrix).unwrap();
        let (re, _) = back.element(0, 0).unwrap();
        assert!((re - 1.0).abs() < 1e-15);
    }

    #[cfg(feature = "entanglement")]
    #[test]
    fn test_eigenvalues_pure_state() {
        let amps = vec![(1.0, 0.0), (0.0, 0.0)];
        let dm = crate::entanglement::DensityMatrix::from_pure_state(&amps);
        let evals = eigenvalues_hermitian(&dm).unwrap();
        // Pure state: one eigenvalue ≈ 1, rest ≈ 0
        let max_eval = evals.iter().cloned().fold(0.0_f64, f64::max);
        assert!((max_eval - 1.0).abs() < 1e-10);
    }

    #[cfg(feature = "operator")]
    #[test]
    fn test_commutator_pauli() {
        // [X, Y] = 2iZ
        let x = crate::operator::Operator::pauli_x();
        let y = crate::operator::Operator::pauli_y();
        let comm = commutator(&x, &y).unwrap();
        let z = crate::operator::Operator::pauli_z();
        for i in 0..2 {
            for j in 0..2 {
                let (c_re, c_im) = comm.element(i, j).unwrap();
                let (z_re, z_im) = z.element(i, j).unwrap();
                // [X,Y] = 2iZ: (re, im) → 2*(-z_im, z_re)
                assert!((c_re - 2.0 * (-z_im)).abs() < 1e-10);
                assert!((c_im - 2.0 * z_re).abs() < 1e-10);
            }
        }
    }
}
