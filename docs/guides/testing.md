# Kana — Testing Guide

## Running Tests

```bash
# Full suite (all features)
cargo test --all-features

# Default features only
cargo test

# Specific module
cargo test --all-features state::tests
cargo test --all-features operator::tests
cargo test --all-features entanglement::tests
cargo test --all-features circuit::tests
cargo test --all-features dynamics::tests
cargo test --all-features safe::tests
cargo test --all-features bridge::tests
cargo test --all-features parallel::tests

# Doc tests (includes compile_fail)
cargo test --all-features --doc

# Integration tests
cargo test --all-features --test integration
```

## Test Categories

### Unit Tests (per module)
- Gate correctness (Pauli algebra, involutions, controlled gates)
- Measurement determinism (fixed random values → predictable outcomes)
- Entanglement measures (Bell states, product states, known analytical values)
- Decomposition roundtrips (ZYZ → reconstruct → compare)
- Noise channel properties (trace preservation, purity reduction)
- Time evolution (unitarity, energy conservation, decay rates)

### Integration Tests (`tests/integration.rs`)
- Cross-module workflows (circuit → state → entanglement)
- Algorithm verification (Deutsch-Jozsa, Grover, QFT roundtrip)
- Noise on circuit outputs
- Bloch sphere ↔ tomography roundtrip

### Compile-Time Tests
- `safe::Qubit` no-cloning: `compile_fail` doctest verifies moved qubit cannot be reused
- Send+Sync assertions on all public types

### Property Tests (recommended additions)
- Unitarity: `U†U = I` for all gate constructors
- Normalization: `||ψ|| = 1` preserved through circuits
- Trace preservation: `Tr(E(ρ)) = 1` for all noise channels

## Writing New Tests

```rust
#[test]
fn test_gate_preserves_normalization() {
    let gate = Operator::hadamard();
    let state = StateVector::zero(1);
    let result = gate.apply(&state).unwrap();
    assert!((result.norm() - 1.0).abs() < NORM_TOLERANCE);
}
```

## Benchmarks

```bash
# Full benchmark suite
cargo bench --all-features

# Specific group
cargo bench --all-features -- "circuit/"
cargo bench --all-features -- "entanglement/"

# Quick comparison
cargo bench --all-features -- --quick
```

### Benchmark Groups
- `state/` — creation, inner product, tensor product, probabilities
- `operator/` — gate creation, apply, multiply, dagger, tensor product
- `entanglement/` — Bell states, concurrence, density matrix, entropy
- `circuit/` — execution (1-3q), algorithms (QFT, Grover, VQE), fusion
- `noise/` — depolarizing, amplitude damping, phase damping
- `measurement/` — measure, measure_qubit, sample, Bloch vector
- `rotation/` — Rx creation, apply
