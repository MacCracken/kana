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
- **entanglement**: State fidelity, trace distance, partial transpose
- **entanglement**: Negativity, log-negativity, mutual information, entanglement of formation
- **entanglement**: Choi matrix representation, process fidelity, average gate fidelity
- **entanglement**: DensityMatrix::validate (Hermitian, PSD, trace-1 checks)
- **entanglement**: Kraus completeness validation in NoiseChannel::new
- **operator**: ZYZ Euler decomposition and reconstruction
- **operator**: KAK entangling structure estimation for two-qubit gates
- **operator**: SparseOperator (COO format) with apply, multiply, tensor_product
- **operator**: Rotation gates Rx, Ry, Rz, arbitrary Phase
- **safe**: QuantumBuilder — ownership-based circuit builder with compile-time no-cloning enforcement
- **safe**: Qubit (move-only), ClassicalBit types
- **dynamics**: Hamiltonian with optional Lindblad dissipators
- **dynamics**: Schrodinger evolution via matrix exponential
- **dynamics**: Lindblad master equation evolution (RK4 integrator)
- **dynamics**: expectation_value function
- **bridge**: Hisab interop (Complex, ComplexMatrix conversions, kronecker, eigenvalues, matrix_exp, commutator, anticommutator)
- **parallel**: Rayon-parallelized gate application (1/2/3-qubit, auto-threshold)
- **parallel**: Parallel Born-rule sampling
- **circuit**: Direct statevector simulation (10-58x faster than matrix expansion)
- **circuit**: Gate fusion and inverse cancellation in optimize()
- **circuit**: Periodic renormalization during long circuits
- **circuit**: Arbitrary basis measurement
- **state**: try_zero with OOM handling, most_probable, support_size, renormalize
- Prelude module for convenient imports
- Send+Sync compile-time trait assertions on all public types
- NORM_TOLERANCE constant for floating-point comparisons

### Changed

- Von Neumann entropy uses 2n×2n real embedding Jacobi (handles complex Hermitian matrices)
- `operator::apply()` uses direct slice access (no per-element bounds checking)
- `ai::register_agent()` sends api_key as Bearer token, checks HTTP status
- Lindblad integrator upgraded from Euler to RK4
- MAX_QUBITS raised from 24 to 28 (4 GiB state vectors)
- License identifier updated to SPDX `GPL-3.0-only`
- hisab dependency updated to 1.4.0 (optional, feature-gated)
- tokio features trimmed from "full" to "rt-multi-thread, macros"
- Feature gates: circuit implies state+operator, entanglement implies state

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
