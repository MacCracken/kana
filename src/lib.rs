//! Kana — Quantum mechanics simulation for AGNOS
//!
//! Sanskrit: कण (kana) — particle, atom
//!
//! Provides quantum state vectors, Hilbert spaces, unitary operators,
//! entanglement simulation, and quantum circuit primitives. Built on
//! [hisab](https://crates.io/crates/hisab) for complex linear algebra
//! and tensor products.
//!
//! # Modules
//!
//! - [`state`] — State vectors, kets/bras, Hilbert spaces, superposition
//! - [`operator`] — Unitary operators, observables, Pauli matrices, measurement
//! - [`entanglement`] — Bell states, density matrices, partial trace, concurrence
//! - [`circuit`] — Quantum gates, circuit construction, measurement
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

#[cfg(feature = "logging")]
pub mod logging;

#[cfg(feature = "ai")]
pub mod ai;

#[cfg(feature = "hisab-bridge")]
pub mod bridge;

pub use error::{KanaError, Result};

#[cfg(feature = "state")]
pub use state::{NORM_TOLERANCE, StateVector};

#[cfg(feature = "operator")]
pub use operator::Operator;

#[cfg(feature = "entanglement")]
pub use entanglement::{DensityMatrix, NoiseChannel};

#[cfg(feature = "circuit")]
pub use circuit::Circuit;
