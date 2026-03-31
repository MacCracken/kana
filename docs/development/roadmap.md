# Kana — Development Roadmap

> Quantum mechanics simulation for AGNOS

## V0.1 — Foundation ✓

- [x] State vectors (computational basis, superposition, tensor products)
- [x] Quantum operators (Pauli, Hadamard, S, T, identity)
- [x] Operator algebra (multiply, dagger, tensor product)
- [x] Entanglement (Bell states, concurrence, density matrices)
- [x] Partial trace for bipartite systems
- [x] Von Neumann entropy
- [x] Quantum circuits (single-qubit gates)
- [x] Error types, logging, AI integration scaffold

## V0.2 — Multi-Qubit Gates & Measurement ✓

- [x] CNOT, CZ, SWAP gates
- [x] Toffoli (CCX), Fredkin (CSWAP) gates
- [x] Controlled-U (arbitrary controlled gate)
- [x] Measurement (projective, computational basis, single-qubit)
- [x] Born rule sampling
- [x] Circuit visualization (text-based)

## V0.3 — Advanced State Operations ✓

- [x] Schmidt decomposition
- [x] Quantum teleportation protocol
- [x] Superdense coding
- [x] Bloch sphere representation (angles + vector)
- [x] State tomography helpers

## V0.4 — Quantum Algorithms ✓

- [x] Deutsch-Jozsa algorithm
- [x] Grover's search
- [x] Quantum Fourier transform
- [x] Inverse QFT
- [x] VQE ansatz (hardware-efficient)
- [x] Rotation gates (Rx, Ry, Rz, Phase)

## V0.5 — Noise & Decoherence ✓

- [x] Kraus operators (general noise channels)
- [x] Depolarizing channel
- [x] Amplitude damping
- [x] Phase damping
- [x] Noise model composition

## V1.0 — Stable Release

- [x] API review and stabilization
- [x] Feature audit
- [x] Complete test coverage (139+ tests)
- [x] Performance benchmarks (35+ benchmarks)
- [x] Cleanliness: fmt, clippy, audit, deny all clean

## Post-V1 — Performance & Scale

- [x] Statevector simulation via index permutation (1/2/3-qubit direct application)
- [ ] Sparse matrix representation for large qubit systems
- [x] Gate fusion / circuit optimization pass (`optimize()` fuses adjacent 1q gates)
- [ ] Parallel operator application (rayon)
- [ ] Memory-mapped state vectors for 16+ qubit systems
- [ ] Profile-guided optimization of hot paths (apply, tensor_product, expand_gate)

## Consumer Mapping

| Consumer | Uses |
|----------|------|
| joshua | Quantum simulation mode, agent quantum states |
| kiran | Quantum-aware game mechanics (via joshua) |

## Boundaries

| Domain | Owned by | NOT kana |
|--------|----------|----------|
| Complex linear algebra | hisab | kana uses hisab's num module |
| Quantum simulation scheduling | joshua | kana provides primitives only |
| Quantum chemistry | future crate | kana is general QM, not molecular |
