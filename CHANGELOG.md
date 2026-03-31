# Changelog

All notable changes to kana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-03-24

### Added

- **state**: State vectors, |0⟩/|1⟩/|+⟩/|−⟩ basis states, superposition, tensor products, inner products, normalization validation
- **operator**: Quantum operators, Pauli-X/Y/Z, Hadamard, S/T phase gates, identity, dagger, multiply, tensor product, apply to state
- **entanglement**: Density matrices from pure states, trace, purity, von Neumann entropy, partial trace, Bell states (Φ±, Ψ±), concurrence
- **circuit**: Quantum circuits, gate sequences, single-qubit gate expansion, execution on initial states
- **error**: KanaError with domain-specific variants (DimensionMismatch, NotNormalized, NotUnitary, InvalidQubitIndex, IncompatibleSubsystems)
- **ai**: Daimon/hoosh client integration (feature-gated)
- **logging**: Structured logging via KANA_LOG (feature-gated)
- Infrastructure: CI/CD, deny.toml, codecov, benchmarks, Makefile
