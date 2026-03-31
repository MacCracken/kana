# Kana — Quantum Mechanics Simulation

> Sanskrit: कण (kana) — particle, atom

Quantum mechanics simulation for the AGNOS ecosystem. State vectors, Hilbert spaces, unitary operators, entanglement, and quantum circuit primitives.

Built on [hisab](https://crates.io/crates/hisab) for complex linear algebra and tensor products.

## Quick Start

```rust
use kana::state::StateVector;
use kana::operator::Operator;
use kana::entanglement;

// Create |0⟩ and apply Hadamard
let zero = StateVector::zero(1);
let h = Operator::hadamard();
let plus = h.apply(&zero).unwrap();
assert!((plus.probability(0).unwrap() - 0.5).abs() < 1e-10);

// Bell state entanglement
let bell = entanglement::bell_phi_plus();
let c = entanglement::concurrence_pure(&bell);
assert!((c - 1.0).abs() < 1e-10); // maximally entangled
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `state` | Yes | State vectors, kets/bras, Hilbert spaces |
| `operator` | Yes | Unitary operators, Pauli matrices, measurement |
| `entanglement` | Yes | Bell states, density matrices, partial trace |
| `circuit` | Yes | Quantum gates, circuit construction |
| `ai` | No | Daimon/hoosh AI integration |
| `logging` | No | Structured logging via `KANA_LOG` |

## Architecture

```
kana
├── state         — StateVector, superposition, tensor products
├── operator      — Operator, Pauli gates, Hadamard, apply
├── entanglement  — DensityMatrix, Bell states, concurrence
├── circuit       — Circuit, Gate, execution
├── error         — KanaError
├── ai            — DaimonClient (optional)
└── logging       — Structured logging (optional)
```

## Consumers

- **joshua** — quantum simulation mode
- **kiran** — quantum-aware game mechanics (via joshua)

## Building

```bash
cargo build                    # default features
cargo build --all-features     # everything
cargo test --all-features      # full test suite
make check                     # fmt + clippy + test + audit
make bench                     # criterion benchmarks with history
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
