//! Quantum circuits — gate sequences, circuit construction, measurement.
//!
//! A quantum circuit is a sequence of gates applied to qubits.

use std::fmt;

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
    /// Note: not serialized. Deserialized circuits must be rebuilt via constructors.
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

    /// Add a two-qubit gate (e.g. CNOT, CZ, SWAP).
    ///
    /// `targets` should be `[control, target]` for controlled gates.
    pub fn add_two_qubit_gate(
        &mut self,
        name: &str,
        targets: [usize; 2],
        operator: Operator,
    ) -> Result<()> {
        for &t in &targets {
            if t >= self.num_qubits {
                return Err(KanaError::InvalidQubitIndex {
                    index: t,
                    num_qubits: self.num_qubits,
                });
            }
        }
        if targets[0] == targets[1] {
            return Err(KanaError::InvalidParameter {
                reason: "two-qubit gate targets must be distinct".into(),
            });
        }
        if operator.dim() != 4 {
            return Err(KanaError::DimensionMismatch {
                expected: 4,
                got: operator.dim(),
            });
        }
        self.gates.push(Gate {
            name: name.to_string(),
            targets: targets.to_vec(),
            operator: Some(operator),
        });
        Ok(())
    }

    /// Add a CNOT gate (control → target).
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<()> {
        self.add_two_qubit_gate("CNOT", [control, target], Operator::cnot())
    }

    /// Add a CZ gate.
    pub fn cz(&mut self, qubit_a: usize, qubit_b: usize) -> Result<()> {
        self.add_two_qubit_gate("CZ", [qubit_a, qubit_b], Operator::cz())
    }

    /// Add a SWAP gate.
    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) -> Result<()> {
        self.add_two_qubit_gate("SWAP", [qubit_a, qubit_b], Operator::swap())
    }

    /// Add a controlled-U gate from an arbitrary single-qubit operator.
    pub fn controlled_u(&mut self, control: usize, target: usize, u: &Operator) -> Result<()> {
        let cu = Operator::controlled(u)?;
        self.add_two_qubit_gate("CU", [control, target], cu)
    }

    /// Add a three-qubit gate (e.g. Toffoli, Fredkin).
    ///
    /// `targets` should be `[q0, q1, q2]` in the gate's qubit order.
    pub fn add_three_qubit_gate(
        &mut self,
        name: &str,
        targets: [usize; 3],
        operator: Operator,
    ) -> Result<()> {
        for &t in &targets {
            if t >= self.num_qubits {
                return Err(KanaError::InvalidQubitIndex {
                    index: t,
                    num_qubits: self.num_qubits,
                });
            }
        }
        if targets[0] == targets[1] || targets[0] == targets[2] || targets[1] == targets[2] {
            return Err(KanaError::InvalidParameter {
                reason: "three-qubit gate targets must be distinct".into(),
            });
        }
        if operator.dim() != 8 {
            return Err(KanaError::DimensionMismatch {
                expected: 8,
                got: operator.dim(),
            });
        }
        self.gates.push(Gate {
            name: name.to_string(),
            targets: targets.to_vec(),
            operator: Some(operator),
        });
        Ok(())
    }

    /// Add a Toffoli (CCX) gate: flips q2 iff q0 and q1 are both |1⟩.
    pub fn toffoli(&mut self, control_a: usize, control_b: usize, target: usize) -> Result<()> {
        self.add_three_qubit_gate(
            "Toffoli",
            [control_a, control_b, target],
            Operator::toffoli(),
        )
    }

    /// Add a Fredkin (CSWAP) gate: swaps q1 and q2 iff q0 is |1⟩.
    pub fn fredkin(&mut self, control: usize, target_a: usize, target_b: usize) -> Result<()> {
        self.add_three_qubit_gate(
            "Fredkin",
            [control, target_a, target_b],
            Operator::fredkin(),
        )
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

    /// Add Phase-S gate to a qubit.
    pub fn phase_s(&mut self, target: usize) -> Result<()> {
        self.add_gate("S", target, Operator::phase_s())
    }

    /// Add T gate to a qubit.
    pub fn phase_t(&mut self, target: usize) -> Result<()> {
        self.add_gate("T", target, Operator::phase_t())
    }

    /// Build a quantum teleportation circuit.
    ///
    /// Uses 3 qubits: q0 = state to teleport, q1 = Alice's Bell pair half,
    /// q2 = Bob's Bell pair half.
    ///
    /// The circuit:
    /// 1. Create Bell pair between q1, q2
    /// 2. Alice: CNOT(q0, q1), then H(q0)
    /// 3. Measure q0, q1
    /// 4. Bob: conditional X and Z corrections (classically controlled)
    ///
    /// After measurement, Bob's qubit q2 holds the teleported state
    /// (up to X/Z corrections determined by measurement outcomes).
    #[must_use]
    pub fn teleportation() -> Self {
        let mut c = Self::new(3);
        // Create Bell pair: q1, q2
        c.hadamard(1).unwrap();
        c.cnot(1, 2).unwrap();
        // Alice's operations on q0, q1
        c.cnot(0, 1).unwrap();
        c.hadamard(0).unwrap();
        // Measurements
        c.measure(0).unwrap();
        c.measure(1).unwrap();
        c
    }

    /// Apply teleportation corrections to Bob's qubit based on measurement results.
    ///
    /// `m0` = measurement of q0, `m1` = measurement of q1.
    /// Appends X and/or Z gates to qubit 2 as needed.
    pub fn teleportation_correction(&mut self, m0: usize, m1: usize) -> Result<()> {
        if m1 == 1 {
            self.pauli_x(2)?;
        }
        if m0 == 1 {
            self.pauli_z(2)?;
        }
        Ok(())
    }

    /// Build a superdense coding circuit.
    ///
    /// Uses 2 qubits. Alice encodes 2 classical bits (b0, b1) into one qubit
    /// of a shared Bell pair. Bob decodes by reversing the Bell state.
    ///
    /// Encoding: b1=1 → X, b0=1 → Z on Alice's qubit (q0).
    /// Decoding: CNOT(q0,q1), H(q0), then measure both.
    pub fn superdense_coding(b0: bool, b1: bool) -> Self {
        let mut c = Self::new(2);
        // Create Bell pair
        c.hadamard(0).unwrap();
        c.cnot(0, 1).unwrap();
        // Alice encodes
        if b1 {
            c.pauli_x(0).unwrap();
        }
        if b0 {
            c.pauli_z(0).unwrap();
        }
        // Bob decodes
        c.cnot(0, 1).unwrap();
        c.hadamard(0).unwrap();
        // Measure
        c.measure(0).unwrap();
        c.measure(1).unwrap();
        c
    }

    /// Add rotation gates.
    pub fn rx(&mut self, target: usize, theta: f64) -> Result<()> {
        self.add_gate("Rx", target, Operator::rx(theta))
    }

    pub fn ry(&mut self, target: usize, theta: f64) -> Result<()> {
        self.add_gate("Ry", target, Operator::ry(theta))
    }

    pub fn rz(&mut self, target: usize, theta: f64) -> Result<()> {
        self.add_gate("Rz", target, Operator::rz(theta))
    }

    /// Build a Deutsch-Jozsa circuit for n input qubits.
    ///
    /// The oracle is specified as a function: `oracle(circuit, input_qubits, output_qubit)`.
    /// Total qubits = n + 1 (n input + 1 output).
    ///
    /// After execution, measuring the input qubits gives:
    /// - all 0s → constant function
    /// - any non-zero → balanced function
    pub fn deutsch_jozsa<F>(n: usize, oracle: F) -> Self
    where
        F: FnOnce(&mut Self, &[usize], usize),
    {
        let total = n + 1;
        let mut c = Self::new(total);
        let output = n;

        // Prepare output qubit in |1⟩
        c.pauli_x(output).unwrap();

        // Apply H to all qubits
        for q in 0..total {
            c.hadamard(q).unwrap();
        }

        // Apply oracle
        let input_qubits: Vec<usize> = (0..n).collect();
        oracle(&mut c, &input_qubits, output);

        // Apply H to input qubits
        for q in 0..n {
            c.hadamard(q).unwrap();
        }

        // Measure input qubits
        for q in 0..n {
            c.measure(q).unwrap();
        }
        c
    }

    /// Build a Grover's search circuit for n qubits.
    ///
    /// `oracle`: marks target states by flipping their phase.
    /// `iterations`: number of Grover iterations (optimal ≈ π/4 · √(2^n)).
    ///
    /// For n > 3, uses ancilla qubits for the multi-controlled-Z decomposition.
    /// Total qubits = n + max(0, n-3) ancillas.
    pub fn grover<F>(n: usize, iterations: usize, oracle: F) -> Self
    where
        F: Fn(&mut Self, &[usize]),
    {
        let n_ancilla = n.saturating_sub(3);
        let total_qubits = n + n_ancilla;
        let mut c = Self::new(total_qubits);
        let qubits: Vec<usize> = (0..n).collect();

        // Initial superposition on data qubits
        for &q in &qubits {
            c.hadamard(q).unwrap();
        }

        for _ in 0..iterations {
            // Oracle
            oracle(&mut c, &qubits);

            // Diffusion operator: 2|s⟩⟨s| − I where |s⟩ = H|0⟩^⊗n
            // = H^⊗n (2|0⟩⟨0| − I) H^⊗n
            for &q in &qubits {
                c.hadamard(q).unwrap();
            }
            for &q in &qubits {
                c.pauli_x(q).unwrap();
            }
            // Multi-controlled Z: phase flip |11...1⟩
            Self::multi_controlled_z(&mut c, &qubits, n);
            for &q in &qubits {
                c.pauli_x(q).unwrap();
            }
            for &q in &qubits {
                c.hadamard(q).unwrap();
            }
        }

        // Measure data qubits only
        for &q in &qubits {
            c.measure(q).unwrap();
        }
        c
    }

    /// Append a multi-controlled-Z gate on the given qubits.
    ///
    /// For n=1: Z gate. For n=2: CZ. For n=3: H-Toffoli-H.
    /// For n>3: Toffoli cascade with ancilla qubits starting at index n.
    fn multi_controlled_z(c: &mut Self, qubits: &[usize], n: usize) {
        if n == 1 {
            c.pauli_z(qubits[0]).unwrap();
        } else if n == 2 {
            c.cz(qubits[0], qubits[1]).unwrap();
        } else {
            let last = qubits[n - 1];
            c.hadamard(last).unwrap();
            if n == 3 {
                c.toffoli(qubits[0], qubits[1], last).unwrap();
            } else {
                // Toffoli cascade: use ancilla qubits at indices n, n+1, ...
                // Forward pass: reduce n controls to 1 using ancillas
                let ancilla_start = n; // ancilla qubit indices in the circuit
                // First Toffoli: controls[0], controls[1] → ancilla[0]
                c.toffoli(qubits[0], qubits[1], ancilla_start).unwrap();
                // Subsequent Toffolis: controls[i], ancilla[i-2] → ancilla[i-1]
                for (idx, &q) in qubits[2..(n - 1)].iter().enumerate() {
                    c.toffoli(q, ancilla_start + idx, ancilla_start + idx + 1)
                        .unwrap();
                }
                // Final CNOT: last ancilla → target (last data qubit)
                c.cnot(ancilla_start + n - 3, last).unwrap();
                // Reverse pass: uncompute ancillas
                for (idx, &q) in qubits[2..(n - 1)].iter().enumerate().rev() {
                    c.toffoli(q, ancilla_start + idx, ancilla_start + idx + 1)
                        .unwrap();
                }
                c.toffoli(qubits[0], qubits[1], ancilla_start).unwrap();
            }
            c.hadamard(last).unwrap();
        }
    }

    /// Build a Quantum Fourier Transform circuit on n qubits.
    ///
    /// Applies the QFT: |j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩
    pub fn qft(n: usize) -> Self {
        let mut c = Self::new(n);
        for j in 0..n {
            c.hadamard(j).unwrap();
            for k in (j + 1)..n {
                let angle = std::f64::consts::PI / (1 << (k - j)) as f64;
                let cp = Operator::controlled(&Operator::phase(angle)).unwrap();
                c.add_two_qubit_gate("CP", [k, j], cp).unwrap();
            }
        }
        // Swap qubits to reverse order (standard QFT convention)
        for i in 0..n / 2 {
            c.swap(i, n - 1 - i).unwrap();
        }
        c
    }

    /// Build an inverse QFT circuit on n qubits.
    pub fn inverse_qft(n: usize) -> Self {
        let mut c = Self::new(n);
        // Reverse swap
        for i in 0..n / 2 {
            c.swap(i, n - 1 - i).unwrap();
        }
        for j in (0..n).rev() {
            for k in ((j + 1)..n).rev() {
                let angle = -std::f64::consts::PI / (1 << (k - j)) as f64;
                let cp = Operator::controlled(&Operator::phase(angle)).unwrap();
                c.add_two_qubit_gate("CP", [k, j], cp).unwrap();
            }
            c.hadamard(j).unwrap();
        }
        c
    }

    /// Build a simple VQE ansatz circuit (hardware-efficient ansatz).
    ///
    /// `params` is a flat array of rotation angles: for each layer,
    /// each qubit gets (Ry, Rz), then CNOT entanglement between adjacent pairs.
    /// Total params needed: n_qubits * 2 * n_layers.
    pub fn vqe_ansatz(n_qubits: usize, n_layers: usize, params: &[f64]) -> Result<Self> {
        let params_needed = n_qubits * 2 * n_layers;
        if params.len() != params_needed {
            return Err(KanaError::InvalidParameter {
                reason: format!(
                    "VQE ansatz needs {} params ({} qubits × 2 × {} layers), got {}",
                    params_needed,
                    n_qubits,
                    n_layers,
                    params.len()
                ),
            });
        }
        let mut c = Self::new(n_qubits);
        let mut idx = 0;
        for _layer in 0..n_layers {
            // Rotation layer
            for q in 0..n_qubits {
                c.ry(q, params[idx])?;
                c.rz(q, params[idx + 1])?;
                idx += 2;
            }
            // Entanglement layer
            for q in 0..n_qubits.saturating_sub(1) {
                c.cnot(q, q + 1)?;
            }
        }
        Ok(c)
    }

    /// Add a measurement marker on a qubit.
    ///
    /// When the circuit is executed with `execute_with_measurement`,
    /// this causes projective measurement and state collapse at this point.
    pub fn measure(&mut self, target: usize) -> Result<()> {
        if target >= self.num_qubits {
            return Err(KanaError::InvalidQubitIndex {
                index: target,
                num_qubits: self.num_qubits,
            });
        }
        self.gates.push(Gate {
            name: "M".to_string(),
            targets: vec![target],
            operator: None,
        });
        Ok(())
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
            if gate.name == "M" {
                continue; // measurement gates are no-ops in non-measurement execute
            }
            let op = gate.operator.as_ref().ok_or_else(|| {
                KanaError::InvalidParameter {
                    reason: format!(
                        "gate '{}' has no operator (deserialized circuits cannot be executed directly)",
                        gate.name
                    ),
                }
            })?;
            let full_op = self.expand_gate(op, &gate.targets)?;
            state = full_op.apply(&state)?;
        }
        Ok(state)
    }

    /// Execute the circuit with measurement, using provided random values.
    ///
    /// Returns `(final_state, measurement_results)` where measurement_results
    /// contains `(qubit_index, bit_value)` for each measurement gate in order.
    pub fn execute_with_measurement(
        &self,
        random_values: &[f64],
    ) -> Result<(StateVector, Vec<(usize, usize)>)> {
        self.execute_on_with_measurement(StateVector::zero(self.num_qubits), random_values)
    }

    /// Execute the circuit on a given state with measurement.
    pub fn execute_on_with_measurement(
        &self,
        mut state: StateVector,
        random_values: &[f64],
    ) -> Result<(StateVector, Vec<(usize, usize)>)> {
        if state.num_qubits() != self.num_qubits {
            return Err(KanaError::DimensionMismatch {
                expected: 1 << self.num_qubits,
                got: state.dimension(),
            });
        }
        let mut measurements = Vec::new();
        let mut r_idx = 0;
        for gate in &self.gates {
            if gate.name == "M" {
                let r = random_values.get(r_idx).copied().ok_or_else(|| {
                    KanaError::InvalidParameter {
                        reason: "not enough random values for measurements".into(),
                    }
                })?;
                r_idx += 1;
                let target = gate.targets[0];
                let (bit, collapsed) = state.measure_qubit(target, r)?;
                state = collapsed;
                measurements.push((target, bit));
            } else {
                let op = gate.operator.as_ref().ok_or_else(|| {
                    KanaError::InvalidParameter {
                        reason: format!(
                            "gate '{}' has no operator (deserialized circuits cannot be executed directly)",
                            gate.name
                        ),
                    }
                })?;
                let full_op = self.expand_gate(op, &gate.targets)?;
                state = full_op.apply(&state)?;
            }
        }
        Ok((state, measurements))
    }

    /// Expand a gate to the full n-qubit Hilbert space.
    ///
    /// For single-qubit gates: I ⊗ ... ⊗ U ⊗ ... ⊗ I
    /// For two-qubit gates on adjacent qubits: I ⊗ ... ⊗ U₄ ⊗ ... ⊗ I
    /// For two-qubit gates on non-adjacent qubits: uses SWAP routing.
    fn expand_gate(&self, gate_op: &Operator, targets: &[usize]) -> Result<Operator> {
        match targets.len() {
            1 => self.expand_single_qubit_gate(gate_op, targets[0]),
            2 => self.expand_two_qubit_gate(gate_op, targets[0], targets[1]),
            3 => self.expand_three_qubit_gate(gate_op, targets),
            _ => Err(KanaError::InvalidParameter {
                reason: format!("{}-qubit gate expansion not supported", targets.len()),
            }),
        }
    }

    /// Expand a single-qubit gate: I ⊗ ... ⊗ U ⊗ ... ⊗ I.
    fn expand_single_qubit_gate(&self, gate_op: &Operator, target: usize) -> Result<Operator> {
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

    /// Expand a two-qubit gate to the full Hilbert space.
    ///
    /// If qubits are adjacent (|q1-q0| == 1), tensor directly.
    /// Otherwise, SWAP-route to bring them adjacent, apply, then SWAP back.
    fn expand_two_qubit_gate(&self, gate_op: &Operator, q0: usize, q1: usize) -> Result<Operator> {
        let n = self.num_qubits;

        // For adjacent qubits in natural order, direct tensor product
        if q1 == q0 + 1 {
            return self.expand_adjacent_two_qubit(gate_op, q0);
        }

        // For reversed adjacent qubits, apply SWAP before and after
        if q0 == q1 + 1 {
            let swap_full = self.expand_adjacent_two_qubit(&Operator::swap(), q1)?;
            let gate_full = self.expand_adjacent_two_qubit(gate_op, q1)?;
            let result = swap_full.multiply(&gate_full)?.multiply(&swap_full)?;
            return Ok(result);
        }

        // For non-adjacent qubits, SWAP-route q1 next to q0, apply, then route back.
        // Build chain of SWAPs to move q1 to position q0+1.
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };

        // Move the higher qubit down to lo+1
        let mut forward_swaps = Vec::new();
        if q0 < q1 {
            // Move q1 down: swap (hi-1,hi), (hi-2,hi-1), ... (lo+1,lo+2)
            for pos in (lo + 1..hi).rev() {
                forward_swaps.push(pos);
            }
        } else {
            // Move q1 up: swap (lo,lo+1), (lo+1,lo+2), ... (hi-2,hi-1)
            // Then the gate acts on (hi-1, hi) but we need q0=hi, q1 moved to hi-1
            for pos in lo..hi - 1 {
                forward_swaps.push(pos);
            }
        }

        let mut full_op = Operator::identity(1 << n);

        // Forward SWAPs
        for &pos in &forward_swaps {
            let swap_op = self.expand_adjacent_two_qubit(&Operator::swap(), pos)?;
            full_op = swap_op.multiply(&full_op)?;
        }

        // Apply gate at adjacent position
        let gate_pos = if q0 < q1 { q0 } else { hi - 1 };
        let gate_full = self.expand_adjacent_two_qubit(gate_op, gate_pos)?;
        full_op = gate_full.multiply(&full_op)?;

        // Reverse SWAPs to restore qubit ordering
        for &pos in forward_swaps.iter().rev() {
            let swap_op = self.expand_adjacent_two_qubit(&Operator::swap(), pos)?;
            full_op = swap_op.multiply(&full_op)?;
        }

        Ok(full_op)
    }

    /// Expand a 4×4 gate acting on adjacent qubits (pos, pos+1) into full space.
    fn expand_adjacent_two_qubit(&self, gate_op: &Operator, pos: usize) -> Result<Operator> {
        let mut full = Operator::identity(1);
        let mut qubit = 0;
        while qubit < self.num_qubits {
            if qubit == pos {
                if qubit == 0 {
                    full = gate_op.clone();
                } else {
                    full = full.tensor_product(gate_op);
                }
                qubit += 2;
            } else {
                let id2 = Operator::identity(2);
                if qubit == 0 {
                    full = id2;
                } else {
                    full = full.tensor_product(&id2);
                }
                qubit += 1;
            }
        }
        Ok(full)
    }

    /// Expand an 8×8 gate acting on adjacent qubits (pos, pos+1, pos+2) into full space.
    fn expand_adjacent_three_qubit(&self, gate_op: &Operator, pos: usize) -> Result<Operator> {
        let mut full = Operator::identity(1);
        let mut qubit = 0;
        while qubit < self.num_qubits {
            if qubit == pos {
                if qubit == 0 {
                    full = gate_op.clone();
                } else {
                    full = full.tensor_product(gate_op);
                }
                qubit += 3;
            } else {
                let id2 = Operator::identity(2);
                if qubit == 0 {
                    full = id2;
                } else {
                    full = full.tensor_product(&id2);
                }
                qubit += 1;
            }
        }
        Ok(full)
    }

    /// Expand a 3-qubit gate to the full Hilbert space via SWAP routing.
    ///
    /// Moves the three target qubits to adjacent positions, applies the gate,
    /// then reverses the permutation.
    fn expand_three_qubit_gate(&self, gate_op: &Operator, targets: &[usize]) -> Result<Operator> {
        let n = self.num_qubits;
        let t = [targets[0], targets[1], targets[2]];

        // Track current positions of our logical qubits via a permutation map.
        // perm[i] = where logical qubit i currently sits physically.
        let mut perm: Vec<usize> = (0..n).collect();

        let mut full_op = Operator::identity(1 << n);

        // Helper: swap adjacent physical positions (p, p+1) and update perm
        let apply_swap = |full: &mut Operator, perm: &mut Vec<usize>, p: usize| -> Result<()> {
            let swap_full = self.expand_adjacent_two_qubit(&Operator::swap(), p)?;
            *full = swap_full.multiply(full)?;
            // Update perm: find which logical qubits are at positions p and p+1
            for slot in perm.iter_mut() {
                if *slot == p {
                    *slot = p + 1;
                } else if *slot == p + 1 {
                    *slot = p;
                }
            }
            Ok(())
        };

        // Move t[0] to some anchor position. We'll anchor at min(t[0], t[1], t[2]).
        let anchor = *t.iter().min().unwrap();

        // Move t[0] to anchor
        while perm[t[0]] > anchor {
            let p = perm[t[0]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[0]] < anchor {
            let p = perm[t[0]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        // Move t[1] to anchor+1
        while perm[t[1]] > anchor + 1 {
            let p = perm[t[1]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[1]] < anchor + 1 {
            let p = perm[t[1]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        // Move t[2] to anchor+2
        while perm[t[2]] > anchor + 2 {
            let p = perm[t[2]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[2]] < anchor + 2 {
            let p = perm[t[2]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        // Apply the 3-qubit gate at (anchor, anchor+1, anchor+2)
        let gate_full = self.expand_adjacent_three_qubit(gate_op, anchor)?;
        full_op = gate_full.multiply(&full_op)?;

        // Reverse: move qubits back to original positions.
        // We need to restore perm to identity. Move in reverse order.
        // Move t[2] back first, then t[1], then t[0].
        let orig = [targets[0], targets[1], targets[2]];

        while perm[t[2]] > orig[2] {
            let p = perm[t[2]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[2]] < orig[2] {
            let p = perm[t[2]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        while perm[t[1]] > orig[1] {
            let p = perm[t[1]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[1]] < orig[1] {
            let p = perm[t[1]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        while perm[t[0]] > orig[0] {
            let p = perm[t[0]] - 1;
            apply_swap(&mut full_op, &mut perm, p)?;
        }
        while perm[t[0]] < orig[0] {
            let p = perm[t[0]];
            apply_swap(&mut full_op, &mut perm, p)?;
        }

        Ok(full_op)
    }
}

impl fmt::Display for Circuit {
    /// Render a text-based circuit diagram.
    ///
    /// ```text
    /// q0: ─H──●──────
    /// q1: ────X──●───
    /// q2: ───────X──M
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.num_qubits == 0 || self.gates.is_empty() {
            for q in 0..self.num_qubits {
                writeln!(f, "q{q}: ─")?;
            }
            return Ok(());
        }

        // Build columns: each gate is one column
        let mut columns: Vec<Vec<String>> = Vec::new();

        for gate in &self.gates {
            let mut col = vec!["──".to_string(); self.num_qubits];

            match gate.targets.len() {
                1 => {
                    let t = gate.targets[0];
                    col[t] = match gate.name.as_str() {
                        "M" => "M─".to_string(),
                        name => name.to_string(),
                    };
                }
                2 => {
                    let (q0, q1) = (gate.targets[0], gate.targets[1]);
                    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
                    match gate.name.as_str() {
                        "CNOT" => {
                            col[gate.targets[0]] = "●─".to_string();
                            col[gate.targets[1]] = "X─".to_string();
                        }
                        "CZ" => {
                            col[gate.targets[0]] = "●─".to_string();
                            col[gate.targets[1]] = "●─".to_string();
                        }
                        "SWAP" => {
                            col[gate.targets[0]] = "×─".to_string();
                            col[gate.targets[1]] = "×─".to_string();
                        }
                        "CU" => {
                            col[gate.targets[0]] = "●─".to_string();
                            col[gate.targets[1]] = "U─".to_string();
                        }
                        name => {
                            col[gate.targets[0]] = name.to_string();
                            col[gate.targets[1]] = name.to_string();
                        }
                    }
                    // Draw vertical connections
                    for slot in col.iter_mut().take(hi).skip(lo + 1) {
                        *slot = "│─".to_string();
                    }
                }
                3 => {
                    let targets = &gate.targets;
                    let lo = *targets.iter().min().unwrap();
                    let hi = *targets.iter().max().unwrap();
                    match gate.name.as_str() {
                        "Toffoli" => {
                            col[targets[0]] = "●─".to_string();
                            col[targets[1]] = "●─".to_string();
                            col[targets[2]] = "X─".to_string();
                        }
                        "Fredkin" => {
                            col[targets[0]] = "●─".to_string();
                            col[targets[1]] = "×─".to_string();
                            col[targets[2]] = "×─".to_string();
                        }
                        name => {
                            for &t in targets {
                                col[t] = name.to_string();
                            }
                        }
                    }
                    for (q, slot) in col.iter_mut().enumerate().take(hi).skip(lo + 1) {
                        if !targets.contains(&q) {
                            *slot = "│─".to_string();
                        }
                    }
                }
                _ => {}
            }
            columns.push(col);
        }

        // Determine label width for alignment
        let label_width = format!("q{}:", self.num_qubits - 1).len();

        for q in 0..self.num_qubits {
            write!(f, "{:>width$} ─", format!("q{q}:"), width = label_width)?;
            for col in &columns {
                write!(f, "{:─<2}─", col[q])?;
            }
            if q < self.num_qubits - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
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

    #[test]
    fn test_cnot_bell_state() {
        // H|0⟩ ⊗ |0⟩ → CNOT → |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let mut c = Circuit::new(2);
        c.hadamard(0).unwrap();
        c.cnot(0, 1).unwrap();
        let result = c.execute().unwrap();
        let p00 = result.probability(0).unwrap();
        let p11 = result.probability(3).unwrap();
        assert!((p00 - 0.5).abs() < 1e-10);
        assert!((p11 - 0.5).abs() < 1e-10);
        // |01⟩ and |10⟩ should be zero
        assert!(result.probability(1).unwrap().abs() < 1e-10);
        assert!(result.probability(2).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_cnot_reversed_targets() {
        // CNOT with control=1, target=0 on |10⟩ → |11⟩
        let mut c = Circuit::new(2);
        c.pauli_x(1).unwrap(); // prepare |01⟩ wait no...
        // |0⟩⊗|0⟩ → X on qubit 1 → |0⟩⊗|1⟩ = |01⟩
        // CNOT(1,0): control=1 is |1⟩, so flip target=0: |01⟩ → |11⟩
        c.cnot(1, 0).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(3).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_non_adjacent_qubits() {
        // 3-qubit circuit: CNOT(0, 2)
        // |000⟩ → H(0) → (|0⟩+|1⟩)/√2 ⊗ |00⟩ → CNOT(0,2)
        // → (|000⟩ + |101⟩)/√2
        let mut c = Circuit::new(3);
        c.hadamard(0).unwrap();
        c.cnot(0, 2).unwrap();
        let result = c.execute().unwrap();
        let p000 = result.probability(0).unwrap(); // |000⟩ = index 0
        let p101 = result.probability(5).unwrap(); // |101⟩ = index 5
        assert!((p000 - 0.5).abs() < 1e-10);
        assert!((p101 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_swap_circuit() {
        // |01⟩ → SWAP → |10⟩
        let mut c = Circuit::new(2);
        c.pauli_x(1).unwrap(); // |00⟩ → |01⟩
        c.swap(0, 1).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(2).unwrap() - 1.0).abs() < 1e-10); // |10⟩
    }

    #[test]
    fn test_cz_circuit() {
        // CZ only applies phase to |11⟩
        // Start with |11⟩, apply CZ, check phase
        let mut c = Circuit::new(2);
        c.pauli_x(0).unwrap();
        c.pauli_x(1).unwrap();
        c.cz(0, 1).unwrap();
        let result = c.execute().unwrap();
        // |11⟩ → −|11⟩, still prob 1 at index 3
        assert!((result.probability(3).unwrap() - 1.0).abs() < 1e-10);
        // Verify it's actually −|11⟩
        let (re, im) = result.amplitude(3).unwrap();
        assert!((re - (-1.0)).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_gate_same_target_rejected() {
        let mut c = Circuit::new(2);
        assert!(c.cnot(0, 0).is_err());
    }

    #[test]
    fn test_two_qubit_gate_oob_rejected() {
        let mut c = Circuit::new(2);
        assert!(c.cnot(0, 5).is_err());
    }

    #[test]
    fn test_toffoli_circuit() {
        // |110⟩ → Toffoli → |111⟩
        let mut c = Circuit::new(3);
        c.pauli_x(0).unwrap();
        c.pauli_x(1).unwrap();
        c.toffoli(0, 1, 2).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(7).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli_no_flip_circuit() {
        // |100⟩ → Toffoli → |100⟩ (only one control is set)
        let mut c = Circuit::new(3);
        c.pauli_x(0).unwrap();
        c.toffoli(0, 1, 2).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(4).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fredkin_circuit() {
        // |101⟩ → Fredkin(0,1,2) → |110⟩
        let mut c = Circuit::new(3);
        c.pauli_x(0).unwrap();
        c.pauli_x(2).unwrap();
        c.fredkin(0, 1, 2).unwrap();
        let result = c.execute().unwrap();
        assert!((result.probability(6).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_controlled_u_circuit() {
        // Controlled-H on |10⟩ → H applied to qubit 1
        let mut c = Circuit::new(2);
        c.pauli_x(0).unwrap(); // |10⟩
        c.controlled_u(0, 1, &Operator::hadamard()).unwrap();
        let result = c.execute().unwrap();
        // Control=1, so H applied: |10⟩ → |1⟩(H|0⟩) = |1⟩(|+⟩) = (|10⟩+|11⟩)/√2
        let p10 = result.probability(2).unwrap();
        let p11 = result.probability(3).unwrap();
        assert!((p10 - 0.5).abs() < 1e-10);
        assert!((p11 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_three_qubit_gate_validation() {
        let mut c = Circuit::new(3);
        // Same targets
        assert!(c.toffoli(0, 0, 1).is_err());
        // OOB
        assert!(c.toffoli(0, 1, 5).is_err());
    }

    #[test]
    fn test_display_bell_circuit() {
        let mut c = Circuit::new(2);
        c.hadamard(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.measure(0).unwrap();
        c.measure(1).unwrap();
        let diagram = format!("{c}");
        assert!(diagram.contains("H─"));
        assert!(diagram.contains("●─"));
        assert!(diagram.contains("X─"));
        assert!(diagram.contains("M─"));
    }

    #[test]
    fn test_display_toffoli_circuit() {
        let mut c = Circuit::new(3);
        c.hadamard(0).unwrap();
        c.toffoli(0, 1, 2).unwrap();
        let diagram = format!("{c}");
        assert!(diagram.contains("●─"));
        assert!(diagram.contains("X─"));
    }

    #[test]
    fn test_circuit_measurement() {
        // Bell state, measure both qubits
        let mut c = Circuit::new(2);
        c.hadamard(0).unwrap();
        c.cnot(0, 1).unwrap();
        c.measure(0).unwrap();
        c.measure(1).unwrap();
        // r=0.3 → qubit 0 measures 0 → qubit 1 should also be 0
        let (_state, results) = c.execute_with_measurement(&[0.3, 0.5]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, results[1].1); // correlated
    }

    #[test]
    fn test_circuit_measurement_insufficient_random() {
        let mut c = Circuit::new(1);
        c.measure(0).unwrap();
        c.measure(0).unwrap();
        // Only one random value for two measurements
        assert!(c.execute_with_measurement(&[0.5]).is_err());
    }

    #[test]
    fn test_deutsch_jozsa_constant() {
        // Constant oracle: f(x) = 0 for all x (do nothing)
        let c = Circuit::deutsch_jozsa(2, |_circuit, _inputs, _output| {
            // f(x) = 0: no gates
        });
        let rs = vec![0.5; 2]; // random values for 2 measurements
        let (_state, results) = c.execute_with_measurement(&rs).unwrap();
        // All inputs should measure 0 (constant)
        for &(_, bit) in &results {
            assert_eq!(bit, 0);
        }
    }

    #[test]
    fn test_deutsch_jozsa_balanced() {
        // Balanced oracle: f(x) = x₀ (parity of first bit)
        // Implement as CNOT(input[0], output)
        let c = Circuit::deutsch_jozsa(2, |circuit, inputs, output| {
            circuit.cnot(inputs[0], output).unwrap();
        });
        let rs = vec![0.5; 2];
        let (_state, results) = c.execute_with_measurement(&rs).unwrap();
        // At least one input should measure 1 (balanced)
        let any_one = results.iter().any(|&(_, bit)| bit == 1);
        assert!(any_one);
    }

    #[test]
    fn test_qft_1qubit() {
        // QFT on 1 qubit = Hadamard
        let c = Circuit::qft(1);
        let result = c.execute().unwrap();
        // QFT|0⟩ = H|0⟩ = |+⟩
        assert!((result.probability(0).unwrap() - 0.5).abs() < 1e-10);
        assert!((result.probability(1).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_qft_inverse_qft_identity() {
        // QFT followed by inverse QFT should give identity
        let qft = Circuit::qft(2);
        let state = qft.execute().unwrap();
        let iqft = Circuit::inverse_qft(2);
        let result = iqft.execute_on(state).unwrap();
        // Should be back to |00⟩
        assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vqe_ansatz_structure() {
        let params = vec![0.1; 4]; // 2 qubits × 2 × 1 layer
        let c = Circuit::vqe_ansatz(2, 1, &params).unwrap();
        assert_eq!(c.num_qubits(), 2);
        // 2 Ry + 2 Rz + 1 CNOT = 5 gates
        assert_eq!(c.num_gates(), 5);
    }

    #[test]
    fn test_vqe_ansatz_param_count() {
        // Wrong param count
        assert!(Circuit::vqe_ansatz(2, 1, &[0.1; 3]).is_err());
    }

    #[test]
    fn test_rotation_gates() {
        // Rx(π) should flip |0⟩ to |1⟩ (up to global phase)
        let rx_pi = Operator::rx(std::f64::consts::PI);
        let state = StateVector::zero(1);
        let result = rx_pi.apply(&state).unwrap();
        assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);

        // Ry(π) should flip |0⟩ to |1⟩
        let ry_pi = Operator::ry(std::f64::consts::PI);
        let result = ry_pi.apply(&state).unwrap();
        assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);

        // Rz(π)|+⟩ should give |−⟩
        let rz_pi = Operator::rz(std::f64::consts::PI);
        let plus = StateVector::plus();
        let result = rz_pi.apply(&plus).unwrap();
        // |−⟩ = (|0⟩ − |1⟩)/√2, probs still 0.5/0.5
        assert!((result.probability(0).unwrap() - 0.5).abs() < 1e-10);
    }
}
