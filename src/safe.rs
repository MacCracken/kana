//! Ownership-based quantum circuit builder — compile-time no-cloning enforcement.
//!
//! The [`Qubit`] type is move-only (no `Clone`, no `Copy`). When a gate consumes
//! a qubit, the Rust borrow checker prevents reuse — enforcing the quantum
//! no-cloning theorem at compile time.
//!
//! # Example
//!
//! ```rust
//! use kana::safe::QuantumBuilder;
//!
//! let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
//! let q0 = b.h(q0);
//! let (q0, q1) = b.cx(q0, q1);
//! let circuit = b.build([q0, q1]);
//! let result = circuit.execute().unwrap();
//! ```
//!
//! Attempting to reuse a consumed qubit is a compile error:
//! ```compile_fail
//! use kana::safe::QuantumBuilder;
//!
//! let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
//! let _q0a = b.h(q0);   // q0 is moved here
//! let _q0b = b.x(q0);   // ERROR: use of moved value `q0`
//! ```

use crate::circuit::Circuit;
use crate::operator::Operator;

/// A quantum bit handle — move-only, enforcing no-cloning at compile time.
///
/// Cannot be cloned, copied, or duplicated. When passed to a gate,
/// ownership transfers and the original binding becomes unusable.
#[derive(Debug)]
pub struct Qubit {
    index: usize,
}

// Deliberately NOT implementing Clone or Copy.

/// A classical measurement result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicalBit {
    qubit_index: usize,
}

impl ClassicalBit {
    /// The qubit index this bit was measured from.
    #[inline]
    #[must_use]
    pub fn qubit(&self) -> usize {
        self.qubit_index
    }
}

/// Ownership-based quantum circuit builder.
///
/// Gates consume and return [`Qubit`] handles, enforcing no-cloning
/// via Rust's move semantics. Call [`build`](Self::build) with all
/// qubits to produce an executable [`Circuit`].
pub struct QuantumBuilder {
    num_qubits: usize,
    circuit: Circuit,
}

impl QuantumBuilder {
    /// Create a builder and N qubit handles.
    ///
    /// ```rust
    /// use kana::safe::QuantumBuilder;
    /// let (mut b, [q0, q1, q2]) = QuantumBuilder::new::<3>();
    /// ```
    #[must_use]
    pub fn new<const N: usize>() -> (Self, [Qubit; N]) {
        let builder = Self {
            num_qubits: N,
            circuit: Circuit::new(N),
        };
        let qubits = std::array::from_fn(|i| Qubit { index: i });
        (builder, qubits)
    }

    /// Hadamard gate. Consumes and returns the qubit.
    pub fn h(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .hadamard(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// Pauli-X gate.
    pub fn x(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .pauli_x(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// Pauli-Y gate.
    pub fn y(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .pauli_y(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// Pauli-Z gate.
    pub fn z(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .pauli_z(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// S (phase) gate.
    pub fn s(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .phase_s(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// T gate.
    pub fn t(&mut self, q: Qubit) -> Qubit {
        self.circuit
            .phase_t(q.index)
            .expect("qubit index valid by construction");
        q
    }

    /// Rx rotation gate.
    pub fn rx(&mut self, q: Qubit, theta: f64) -> Qubit {
        self.circuit
            .rx(q.index, theta)
            .expect("qubit index valid by construction");
        q
    }

    /// Ry rotation gate.
    pub fn ry(&mut self, q: Qubit, theta: f64) -> Qubit {
        self.circuit
            .ry(q.index, theta)
            .expect("qubit index valid by construction");
        q
    }

    /// Rz rotation gate.
    pub fn rz(&mut self, q: Qubit, theta: f64) -> Qubit {
        self.circuit
            .rz(q.index, theta)
            .expect("qubit index valid by construction");
        q
    }

    /// CNOT (controlled-X) gate. Consumes and returns both qubits.
    pub fn cx(&mut self, control: Qubit, target: Qubit) -> (Qubit, Qubit) {
        self.circuit
            .cnot(control.index, target.index)
            .expect("qubit index valid by construction");
        (control, target)
    }

    /// CZ (controlled-Z) gate.
    pub fn cz(&mut self, a: Qubit, b: Qubit) -> (Qubit, Qubit) {
        self.circuit
            .cz(a.index, b.index)
            .expect("qubit index valid by construction");
        (a, b)
    }

    /// SWAP gate.
    pub fn swap(&mut self, a: Qubit, b: Qubit) -> (Qubit, Qubit) {
        self.circuit
            .swap(a.index, b.index)
            .expect("qubit index valid by construction");
        // After swap, the qubit handles swap their logical meaning
        (b, a)
    }

    /// Toffoli (CCX) gate.
    pub fn ccx(&mut self, c0: Qubit, c1: Qubit, target: Qubit) -> (Qubit, Qubit, Qubit) {
        self.circuit
            .toffoli(c0.index, c1.index, target.index)
            .expect("qubit index valid by construction");
        (c0, c1, target)
    }

    /// Fredkin (CSWAP) gate.
    pub fn cswap(&mut self, control: Qubit, a: Qubit, b: Qubit) -> (Qubit, Qubit, Qubit) {
        self.circuit
            .fredkin(control.index, a.index, b.index)
            .expect("qubit index valid by construction");
        (control, b, a) // targets are swapped
    }

    /// Controlled-U gate with arbitrary single-qubit unitary.
    pub fn cu(&mut self, control: Qubit, target: Qubit, u: &Operator) -> (Qubit, Qubit) {
        self.circuit
            .controlled_u(control.index, target.index, u)
            .expect("qubit index valid by construction");
        (control, target)
    }

    /// Measure a qubit. Consumes the qubit, returns a classical bit
    /// and the (collapsed) qubit handle.
    pub fn measure(&mut self, q: Qubit) -> (ClassicalBit, Qubit) {
        self.circuit
            .measure(q.index)
            .expect("qubit index valid by construction");
        (
            ClassicalBit {
                qubit_index: q.index,
            },
            q,
        )
    }

    /// Build the circuit. All qubit handles must be returned.
    ///
    /// The const generic `N` must match the number of qubits allocated.
    /// Returning all qubits proves none were lost or duplicated.
    #[must_use]
    pub fn build<const N: usize>(self, _qubits: [Qubit; N]) -> Circuit {
        assert_eq!(
            N, self.num_qubits,
            "must return all {0} qubits, got {N}",
            self.num_qubits
        );
        self.circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::NORM_TOLERANCE;

    #[test]
    fn test_bell_state() {
        let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
        let q0 = b.h(q0);
        let (q0, q1) = b.cx(q0, q1);
        let circuit = b.build([q0, q1]);
        let result = circuit.execute().unwrap();
        // |Φ+⟩ = (|00⟩ + |11⟩)/√2
        assert!((result.probability(0).unwrap() - 0.5).abs() < NORM_TOLERANCE);
        assert!((result.probability(3).unwrap() - 0.5).abs() < NORM_TOLERANCE);
    }

    #[test]
    fn test_ghz_state() {
        let (mut b, [q0, q1, q2]) = QuantumBuilder::new::<3>();
        let q0 = b.h(q0);
        let (q0, q1) = b.cx(q0, q1);
        let (q0, q2) = b.cx(q0, q2);
        let circuit = b.build([q0, q1, q2]);
        let result = circuit.execute().unwrap();
        assert!((result.probability(0).unwrap() - 0.5).abs() < NORM_TOLERANCE);
        assert!((result.probability(7).unwrap() - 0.5).abs() < NORM_TOLERANCE);
    }

    #[test]
    fn test_measurement() {
        let (mut b, [q0]) = QuantumBuilder::new::<1>();
        let q0 = b.h(q0);
        let (bit, q0) = b.measure(q0);
        assert_eq!(bit.qubit(), 0);
        let circuit = b.build([q0]);
        assert_eq!(circuit.num_gates(), 2); // H + M
    }

    #[test]
    fn test_swap_returns_swapped() {
        let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
        let q0 = b.x(q0); // q0 = |1⟩
        // After swap, the returned handles have swapped logical meaning
        let (q0, q1) = b.swap(q0, q1);
        let circuit = b.build([q0, q1]);
        let result = circuit.execute().unwrap();
        // |10⟩ → SWAP → |01⟩
        assert!((result.probability(1).unwrap() - 1.0).abs() < NORM_TOLERANCE);
    }

    #[test]
    fn test_rotation_gates() {
        let (mut b, [q0]) = QuantumBuilder::new::<1>();
        let q0 = b.rx(q0, std::f64::consts::PI);
        let circuit = b.build([q0]);
        let result = circuit.execute().unwrap();
        // Rx(π)|0⟩ = -i|1⟩
        assert!((result.probability(1).unwrap() - 1.0).abs() < NORM_TOLERANCE);
    }

    #[test]
    fn test_toffoli() {
        let (mut b, [q0, q1, q2]) = QuantumBuilder::new::<3>();
        let q0 = b.x(q0);
        let q1 = b.x(q1);
        let (q0, q1, q2) = b.ccx(q0, q1, q2);
        let circuit = b.build([q0, q1, q2]);
        let result = circuit.execute().unwrap();
        assert!((result.probability(7).unwrap() - 1.0).abs() < NORM_TOLERANCE);
    }

    #[test]
    fn test_controlled_u() {
        let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
        let q0 = b.x(q0); // control = |1⟩
        let (q0, q1) = b.cu(q0, q1, &Operator::hadamard());
        let circuit = b.build([q0, q1]);
        let result = circuit.execute().unwrap();
        // |1⟩ ⊗ H|0⟩ = |1⟩(|+⟩)
        assert!((result.probability(2).unwrap() - 0.5).abs() < NORM_TOLERANCE);
        assert!((result.probability(3).unwrap() - 0.5).abs() < NORM_TOLERANCE);
    }

    // This test MUST NOT compile. It is a compile_fail doctest in the module doc.
    // If someone could clone a qubit, no-cloning would be violated.
    // The following would fail:
    //   let (mut b, [q0]) = QuantumBuilder::new::<1>();
    //   let q0 = b.h(q0);
    //   let q0_again = b.h(q0); // ERROR: use of moved value
}
