//! Quantum circuits — gate sequences, circuit construction, measurement.
//!
//! A quantum circuit is a sequence of gates applied to qubits.

use serde::{Deserialize, Serialize};

use crate::error::{KanaError, Result};
use crate::operator::Operator;
use crate::state::StateVector;

/// A gate applied to specific qubit(s) in a circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    /// Name of the gate (for display/serialization).
    pub name: String,
    /// Target qubit indices.
    pub targets: Vec<usize>,
    /// The operator matrix (for the gate's own Hilbert space).
    #[serde(skip)]
    operator: Option<Operator>,
}

/// A quantum circuit: a sequence of gates on a fixed number of qubits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Number of qubits in the circuit.
    num_qubits: usize,
    /// Ordered list of gates to apply.
    gates: Vec<Gate>,
}

impl Circuit {
    /// Create a new empty circuit for n qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Number of qubits in this circuit.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of gates in this circuit.
    #[inline]
    #[must_use]
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Add a single-qubit gate.
    pub fn add_gate(&mut self, name: &str, target: usize, operator: Operator) -> Result<()> {
        if target >= self.num_qubits {
            return Err(KanaError::InvalidQubitIndex {
                index: target,
                num_qubits: self.num_qubits,
            });
        }
        self.gates.push(Gate {
            name: name.to_string(),
            targets: vec![target],
            operator: Some(operator),
        });
        Ok(())
    }

    /// Add Hadamard gate to a qubit.
    pub fn hadamard(&mut self, target: usize) -> Result<()> {
        self.add_gate("H", target, Operator::hadamard())
    }

    /// Add Pauli-X gate to a qubit.
    pub fn pauli_x(&mut self, target: usize) -> Result<()> {
        self.add_gate("X", target, Operator::pauli_x())
    }

    /// Add Pauli-Y gate to a qubit.
    pub fn pauli_y(&mut self, target: usize) -> Result<()> {
        self.add_gate("Y", target, Operator::pauli_y())
    }

    /// Add Pauli-Z gate to a qubit.
    pub fn pauli_z(&mut self, target: usize) -> Result<()> {
        self.add_gate("Z", target, Operator::pauli_z())
    }

    /// Execute the circuit on the |0...0⟩ initial state.
    pub fn execute(&self) -> Result<StateVector> {
        self.execute_on(StateVector::zero(self.num_qubits))
    }

    /// Execute the circuit on a given initial state.
    pub fn execute_on(&self, mut state: StateVector) -> Result<StateVector> {
        if state.num_qubits() != self.num_qubits {
            return Err(KanaError::DimensionMismatch {
                expected: 1 << self.num_qubits,
                got: state.dimension(),
            });
        }
        for gate in &self.gates {
            if let Some(ref op) = gate.operator {
                let full_op = self.expand_gate(op, &gate.targets)?;
                state = full_op.apply(&state)?;
            }
        }
        Ok(state)
    }

    /// Expand a single-qubit gate to the full n-qubit Hilbert space.
    fn expand_gate(&self, gate_op: &Operator, targets: &[usize]) -> Result<Operator> {
        if targets.len() != 1 {
            return Err(KanaError::InvalidParameter {
                reason: "multi-qubit gate expansion not yet implemented".into(),
            });
        }
        let target = targets[0];
        let mut full = Operator::identity(1);

        for qubit in 0..self.num_qubits {
            let op = if qubit == target {
                gate_op.clone()
            } else {
                Operator::identity(2)
            };
            if qubit == 0 {
                full = op;
            } else {
                full = full.tensor_product(&op);
            }
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_circuit() {
        let c = Circuit::new(3);
        assert_eq!(c.num_qubits(), 3);
        assert_eq!(c.num_gates(), 0);
    }

    #[test]
    fn test_hadamard_circuit() {
        let mut c = Circuit::new(1);
        c.hadamard(0).unwrap();
        let result = c.execute().unwrap();
        let p0 = result.probability(0).unwrap();
        let p1 = result.probability(1).unwrap();
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_x_gate_circuit() {
        let mut c = Circuit::new(1);
        c.pauli_x(0).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_x_identity() {
        let mut c = Circuit::new(1);
        c.pauli_x(0).unwrap();
        c.pauli_x(0).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut c = Circuit::new(2);
        assert!(c.hadamard(5).is_err());
    }

    #[test]
    fn test_multi_qubit_circuit() {
        let mut c = Circuit::new(2);
        c.hadamard(0).unwrap();
        let result = c.execute().unwrap();
        // H|0⟩⊗|0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ = (|00⟩+|10⟩)/√2
        let p00 = result.probability(0).unwrap();
        let p10 = result.probability(2).unwrap();
        assert!((p00 - 0.5).abs() < 1e-10);
        assert!((p10 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gate_count() {
        let mut c = Circuit::new(2);
        c.hadamard(0).unwrap();
        c.pauli_x(1).unwrap();
        c.pauli_z(0).unwrap();
        assert_eq!(c.num_gates(), 3);
    }
}
