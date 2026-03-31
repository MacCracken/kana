//! Basic quantum mechanics with kana.

fn main() {
    // Create basis states
    let zero = kana::state::StateVector::zero(1);
    println!("  |0⟩ probabilities: {:?}", zero.probabilities());

    // Apply Hadamard to create superposition
    let h = kana::operator::Operator::hadamard();
    let plus = h.apply(&zero).unwrap();
    println!("H|0⟩ probabilities: {:?}", plus.probabilities());

    // Bell state entanglement
    let bell = kana::entanglement::bell_phi_plus();
    let concurrence = kana::entanglement::concurrence_pure(&bell);
    println!("|Φ+⟩ concurrence:   {concurrence:.4} (1.0 = maximally entangled)");

    // Simple circuit
    let mut circuit = kana::circuit::Circuit::new(2);
    circuit.hadamard(0).unwrap();
    circuit.pauli_x(1).unwrap();
    let result = circuit.execute().unwrap();
    println!("Circuit result:     {:?}", result.probabilities());
}
