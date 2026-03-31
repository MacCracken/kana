//! Error types for kana.

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KanaError {
    #[error("invalid state dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("state vector not normalized: norm = {norm} (expected 1.0)")]
    NotNormalized { norm: f64 },

    #[error("operator not unitary: U†U deviation = {deviation}")]
    NotUnitary { deviation: f64 },

    #[error("invalid qubit index: {index} (system has {num_qubits} qubits)")]
    InvalidQubitIndex { index: usize, num_qubits: usize },

    #[error("incompatible subsystem dimensions for partial trace")]
    IncompatibleSubsystems,

    #[error("division by zero in quantum calculation: {context}")]
    DivisionByZero { context: String },

    #[error("invalid parameter: {reason}")]
    InvalidParameter { reason: String },
}

pub type Result<T> = std::result::Result<T, KanaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        let e = KanaError::DimensionMismatch {
            expected: 4,
            got: 2,
        };
        let msg = e.to_string();
        assert!(msg.contains("expected 4"));
        assert!(msg.contains("got 2"));
    }

    #[test]
    fn test_not_normalized() {
        let e = KanaError::NotNormalized { norm: 0.95 };
        assert!(e.to_string().contains("0.95"));
    }

    #[test]
    fn test_not_unitary() {
        let e = KanaError::NotUnitary { deviation: 0.01 };
        assert!(e.to_string().contains("0.01"));
    }

    #[test]
    fn test_invalid_qubit() {
        let e = KanaError::InvalidQubitIndex {
            index: 5,
            num_qubits: 3,
        };
        let msg = e.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("3 qubits"));
    }

    #[test]
    fn test_incompatible_subsystems() {
        let e = KanaError::IncompatibleSubsystems;
        assert!(e.to_string().contains("partial trace"));
    }

    #[test]
    fn test_division_by_zero() {
        let e = KanaError::DivisionByZero {
            context: "normalization".into(),
        };
        assert!(e.to_string().contains("normalization"));
    }

    #[test]
    fn test_invalid_parameter() {
        let e = KanaError::InvalidParameter {
            reason: "negative probability".into(),
        };
        assert!(e.to_string().contains("negative probability"));
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<f64> = Ok(1.0);
        assert!(ok.is_ok());
        let err: Result<f64> = Err(KanaError::NotNormalized { norm: 0.5 });
        assert!(err.is_err());
    }

    #[test]
    fn test_error_is_debug() {
        let e = KanaError::NotNormalized { norm: 0.5 };
        let debug = format!("{:?}", e);
        assert!(debug.contains("NotNormalized"));
    }
}
