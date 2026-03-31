# Kana — Architecture Overview

## Module Tree

```
kana/
  state        — StateVector, measurement, Bloch sphere, Born sampling
  operator     — Operator, gates (Pauli, rotation, controlled), decompositions (ZYZ, KAK)
                 SparseOperator (COO format)
  entanglement — DensityMatrix, Bell states, entanglement measures (negativity, concurrence,
                 mutual information), noise channels (Kraus, depolarizing, damping),
                 Schmidt decomposition, tomography, Choi matrix, fidelity metrics
  circuit      — Circuit, Gate, gate expansion, direct statevector simulation,
                 optimization (fusion, cancellation), algorithms (Grover, QFT, VQE),
                 text visualization
  safe         — QuantumBuilder, Qubit (move-only), ClassicalBit — compile-time no-cloning
  dynamics     — Hamiltonian, Schrodinger evolution, Lindblad master equation (RK4),
                 expectation values
  bridge       — Hisab interop (Complex, ComplexMatrix conversions, kronecker, eigenvalues,
                 matrix_exp, commutator, anticommutator)
  parallel     — Rayon-parallelized gate application and sampling
  error        — KanaError (7 variants, non-exhaustive)
  logging      — Structured tracing via KANA_LOG
  ai           — Daimon/hoosh client (feature-gated)
```

## Feature Flags

| Feature | Default | Implies | Description |
|---------|---------|---------|-------------|
| `state` | Yes | — | State vectors, measurement, Bloch sphere |
| `operator` | Yes | `state` | Quantum operators, gates, decompositions |
| `entanglement` | Yes | `state` | Density matrices, noise, entanglement measures |
| `circuit` | Yes | `state`, `operator` | Circuits, algorithms, safe builder |
| `hisab-bridge` | No | — | Hisab linear algebra interop |
| `parallel` | No | — | Rayon parallelism for large systems |
| `ai` | No | — | Daimon/hoosh client |
| `logging` | No | — | Structured tracing |
| `full` | No | all | Everything |

## Design Principles

- **Direct statevector simulation**: Gates applied directly to amplitudes via bit-indexed pairs — O(2^n) per gate, not O(4^n) matrix expansion
- **Compile-time no-cloning**: `safe::Qubit` is move-only; rustc enforces the no-cloning theorem
- **Own the stack**: Complex arithmetic inline, Jacobi eigenvalues self-contained, hisab optional via bridge
- **Feature isolation**: Modules compile independently; `circuit` does not require `entanglement`
- **Numerical discipline**: NORM_TOLERANCE constant, periodic renormalization, RK4 for Lindblad, 2n×2n real embedding for complex Hermitian eigenvalues

## Data Flow

```
User code
  |
  v
QuantumBuilder (safe) ──or──> Circuit (index-based)
  |                              |
  v                              v
Circuit                    execute_on / execute_with_measurement
  |                              |
  v                              v
apply_gate_direct ────────> StateVector (amplitudes modified in-place)
  |                              |
  1q: bit-pair iteration         |
  2q: 4-element groups           |
  3q: 8-element groups           |
  |                              v
  +──> parallel module ───> rayon (if dim >= 1024 and feature enabled)
```

## Consumer Mapping

| Consumer | Uses |
|----------|------|
| joshua | Quantum simulation mode, agent quantum states |
| kiran | Quantum-aware game mechanics (via joshua) |

## Boundaries

| Domain | Owned by | NOT kana |
|--------|----------|----------|
| Complex linear algebra | hisab | kana uses via bridge |
| Quantum simulation scheduling | joshua | kana provides primitives |
| Quantum chemistry | future crate | kana is general QM |
