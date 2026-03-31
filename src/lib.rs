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

pub use error::KanaError;
