# Kana — Development Roadmap

> Quantum mechanics simulation for AGNOS

## V0.1 — Foundation (current)

- [x] State vectors (computational basis, superposition, tensor products)
- [x] Quantum operators (Pauli, Hadamard, S, T, identity)
- [x] Operator algebra (multiply, dagger, tensor product)
- [x] Entanglement (Bell states, concurrence, density matrices)
- [x] Partial trace for bipartite systems
- [x] Von Neumann entropy
- [x] Quantum circuits (single-qubit gates)
- [x] Error types, logging, AI integration scaffold

## V0.2 — Multi-Qubit Gates & Measurement

- [ ] CNOT, SWAP, Toffoli, Fredkin gates
- [ ] Controlled-U (arbitrary controlled gate)
- [ ] Measurement (projective, computational basis)
- [ ] Born rule sampling
- [ ] Circuit visualization (text-based)

## V0.3 — Advanced State Operations

- [ ] Schmidt decomposition
- [ ] Quantum teleportation protocol
- [ ] Superdense coding
- [ ] Bloch sphere representation
- [ ] State tomography helpers

## V0.4 — Quantum Algorithms

- [ ] Deutsch-Jozsa algorithm
- [ ] Grover's search
- [ ] Quantum Fourier transform
- [ ] Phase estimation
- [ ] Variational quantum eigensolver (VQE) primitives

## V0.5 — Noise & Decoherence

- [ ] Kraus operators
- [ ] Depolarizing channel
- [ ] Amplitude damping
- [ ] Phase damping
- [ ] Noise model composition

## V1.0 — Stable Release

- [ ] API review and stabilization
- [ ] Feature audit
- [ ] Complete documentation with physics references
- [ ] Performance optimization pass

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
