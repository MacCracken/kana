# Changelog

All notable changes to kana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — 2026-03-30

### Added

- **operator**: CNOT, CZ, SWAP, Toffoli (CCX), Fredkin (CSWAP) gates
- **operator**: Controlled-U for arbitrary single-qubit gates
- **operator**: Rotation gates Rx(θ), Ry(θ), Rz(θ), Phase(φ)
- **state**: Projective measurement with state collapse (full and single-qubit)
- **state**: Born rule sampling for statistical measurement
- **state**: Bloch sphere representation (angles and vector)
- **circuit**: Two-qubit gate expansion with SWAP routing for non-adjacent qubits
- **circuit**: Three-qubit gate expansion with permutation routing
- **circuit**: Measurement gates with `execute_with_measurement`
- **circuit**: Text-based circuit visualization via `Display` trait
- **circuit**: Quantum teleportation protocol builder
- **circuit**: Superdense coding protocol builder
- **circuit**: Deutsch-Jozsa algorithm builder
- **circuit**: Grover's search algorithm builder
- **circuit**: Quantum Fourier Transform and inverse QFT
- **circuit**: VQE hardware-efficient ansatz builder
- **entanglement**: Schmidt decomposition and Schmidt rank
- **entanglement**: State tomography from Pauli expectations
- **entanglement**: Noise channels via Kraus operators
- **entanglement**: Depolarizing, amplitude damping, phase damping channels
- **entanglement**: Noise model composition
- NORM_TOLERANCE constant for floating-point comparisons

### Changed

- Von Neumann entropy now uses Jacobi eigenvalue algorithm (exact for all sizes)
- `operator::apply()` propagates actual normalization deviation instead of dummy value
- `ai::register_agent()` validates response JSON instead of silent fallback
- License identifier updated to SPDX `GPL-3.0-only`
- hisab dependency updated to 1.4.0

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
