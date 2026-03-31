# Kana — Integration Guide

## Quick Start

```rust
use kana::prelude::*;

// Create a Bell state
let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
let q0 = b.h(q0);
let (q0, q1) = b.cx(q0, q1);
let circuit = b.build([q0, q1]);
let state = circuit.execute().unwrap();
```

## Safe Builder (Recommended)

The `QuantumBuilder` enforces no-cloning at compile time:

```rust
use kana::safe::QuantumBuilder;

let (mut b, [q0, q1, q2]) = QuantumBuilder::new::<3>();
let q0 = b.h(q0);
let (q0, q1) = b.cx(q0, q1);
let (q0, q2) = b.cx(q0, q2);
let circuit = b.build([q0, q1, q2]);
// GHZ state: (|000> + |111>)/sqrt(2)
```

## Index-Based API (Algorithms)

For dynamic qubit indices (algorithms, loops):

```rust
use kana::circuit::Circuit;

let c = Circuit::grover(2, 1, |circuit, qubits| {
    circuit.cz(qubits[0], qubits[1]).unwrap();
});
let (_state, results) = c.execute_with_measurement(&[0.5, 0.5]).unwrap();
```

## Measurement

```rust
use kana::state::StateVector;
use kana::operator::Operator;

let state = StateVector::plus();

// Computational basis
let (bit, collapsed) = state.measure_qubit(0, 0.3).unwrap();

// Arbitrary basis (X basis via Hadamard)
let (bit, collapsed) = state.measure_in_basis(0, &Operator::hadamard(), 0.5).unwrap();

// Born-rule sampling (100 shots)
let rs: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
let outcomes = state.sample(&rs).unwrap();
```

## Noise Simulation

```rust
use kana::entanglement::{DensityMatrix, NoiseChannel};

let dm = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
let channel = NoiseChannel::depolarizing(0.1).unwrap();
let noisy = channel.apply(&dm).unwrap();
assert!(noisy.purity() < 1.0);
```

## Time Evolution

```rust
use kana::dynamics::{Hamiltonian, expectation_value};
use kana::operator::Operator;
use kana::state::StateVector;
use kana::entanglement::DensityMatrix;

// Closed system: Schrodinger
let h = Hamiltonian::new(Operator::pauli_z());
let state = StateVector::plus();
let evolved = h.evolve_state(&state, 1.0).unwrap();

// Open system: Lindblad
let mut h = Hamiltonian::new(Operator::pauli_z());
h.add_dissipator(0.1, Operator::pauli_x());
let rho = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
let evolved = h.evolve_density(&rho, 5.0, 500).unwrap();
```

## Hisab Bridge

```rust
// Feature: hisab-bridge
use kana::bridge;
use kana::operator::Operator;

let h = Operator::hadamard();
assert!(bridge::is_unitary(&h, 1e-10).unwrap());

let x = Operator::pauli_x();
let y = Operator::pauli_y();
let comm = bridge::commutator(&x, &y).unwrap();
// [X, Y] = 2iZ
```

## Circuit Optimization

```rust
use kana::circuit::Circuit;

let mut c = Circuit::new(1);
c.hadamard(0).unwrap();
c.hadamard(0).unwrap(); // HH = I
c.pauli_x(0).unwrap();

let opt = c.optimize();
assert_eq!(opt.num_gates(), 1); // HH cancelled, only X remains
```
