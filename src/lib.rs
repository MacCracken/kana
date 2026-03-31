//! # Kana — Quantum mechanics simulation for AGNOS
//!
//! Sanskrit: कण (kana) — particle, atom
//!
//! Provides quantum state vectors, Hilbert spaces, unitary operators,
//! entanglement simulation, quantum circuit primitives, and time evolution.
//! Built on [hisab](https://crates.io/crates/hisab) for complex linear
//! algebra (via the `hisab-bridge` feature).
//!
//! # Modules
//!
//! - [`state`] — State vectors, kets/bras, measurement, Bloch sphere
//! - [`operator`] — Unitary operators, Pauli/rotation gates, decompositions
//! - [`entanglement`] — Density matrices, entanglement measures, noise channels
//! - [`circuit`] — Quantum circuits, gate optimization, algorithms (Grover, QFT, VQE)
//! - [`safe`] — Ownership-based circuit builder (compile-time no-cloning)
//! - [`dynamics`] — Time evolution (Schrodinger, Lindblad master equation)
//! - [`bridge`] — Hisab interop (feature-gated)
//! - [`parallel`] — Rayon-parallelized operations (feature-gated)
//! - [`error`] — Error types

pub mod error;

#[cfg(feature = "state")]
pub mod state;

#[cfg(feature = "operator")]
pub mod operator;

#[cfg(feature = "entanglement")]
pub mod entanglement;

#[cfg(feature = "circuit")]
pub mod circuit;

#[cfg(feature = "circuit")]
pub mod safe;

#[cfg(all(feature = "state", feature = "operator", feature = "entanglement"))]
pub mod dynamics;

#[cfg(feature = "logging")]
pub mod logging;

#[cfg(feature = "ai")]
pub mod ai;

#[cfg(feature = "hisab-bridge")]
pub mod bridge;

#[cfg(all(feature = "parallel", feature = "state", feature = "operator"))]
pub mod parallel;

pub use error::{KanaError, Result};

#[cfg(feature = "state")]
pub use state::{NORM_TOLERANCE, StateVector};

#[cfg(feature = "operator")]
pub use operator::{KakDecomposition, Operator, SparseOperator};

#[cfg(feature = "entanglement")]
pub use entanglement::{DensityMatrix, NoiseChannel};

#[cfg(feature = "circuit")]
pub use circuit::Circuit;

#[cfg(feature = "circuit")]
pub use safe::{ClassicalBit, QuantumBuilder, Qubit};

#[cfg(all(feature = "state", feature = "operator", feature = "entanglement"))]
pub use dynamics::{Hamiltonian, expectation_value};

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::error::{KanaError, Result};

    #[cfg(feature = "state")]
    pub use crate::state::StateVector;

    #[cfg(feature = "operator")]
    pub use crate::operator::Operator;

    #[cfg(feature = "entanglement")]
    pub use crate::entanglement::{DensityMatrix, NoiseChannel};

    #[cfg(feature = "circuit")]
    pub use crate::circuit::Circuit;

    #[cfg(feature = "circuit")]
    pub use crate::safe::{ClassicalBit, QuantumBuilder, Qubit};

    #[cfg(all(feature = "state", feature = "operator", feature = "entanglement"))]
    pub use crate::dynamics::Hamiltonian;
}

// Compile-time trait assertions: all public types must be Send + Sync.
#[cfg(test)]
mod assert_traits {
    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_sync() {
        _assert_send_sync::<crate::error::KanaError>();

        #[cfg(feature = "state")]
        _assert_send_sync::<crate::state::StateVector>();

        #[cfg(feature = "operator")]
        {
            _assert_send_sync::<crate::operator::Operator>();
            _assert_send_sync::<crate::operator::SparseOperator>();
            _assert_send_sync::<crate::operator::KakDecomposition>();
        }

        #[cfg(feature = "entanglement")]
        {
            _assert_send_sync::<crate::entanglement::DensityMatrix>();
            _assert_send_sync::<crate::entanglement::NoiseChannel>();
        }

        #[cfg(feature = "circuit")]
        {
            _assert_send_sync::<crate::circuit::Circuit>();
            _assert_send_sync::<crate::circuit::Gate>();
            _assert_send_sync::<crate::safe::Qubit>();
            _assert_send_sync::<crate::safe::ClassicalBit>();
            _assert_send_sync::<crate::safe::QuantumBuilder>();
        }

        #[cfg(all(feature = "state", feature = "operator", feature = "entanglement"))]
        _assert_send_sync::<crate::dynamics::Hamiltonian>();
    }
}
