//! Kana v1.0 — quantum mechanics simulation examples.

use kana::circuit::Circuit;
use kana::entanglement;
use kana::operator::Operator;
use kana::state::StateVector;

fn main() {
    // --- State vectors ---
    let zero = StateVector::zero(1);
    println!("  |0⟩ probabilities: {:?}", zero.probabilities());

    let h = Operator::hadamard();
    let plus = h.apply(&zero).unwrap();
    println!("H|0⟩ probabilities: {:?}", plus.probabilities());

    // Bloch sphere
    let (x, y, z) = plus.bloch_vector().unwrap();
    println!("  |+⟩ Bloch vector: ({x:.3}, {y:.3}, {z:.3})");

    // --- Bell state via circuit ---
    let mut bell_circuit = Circuit::new(2);
    bell_circuit.hadamard(0).unwrap();
    bell_circuit.cnot(0, 1).unwrap();
    println!("\nBell circuit:\n{bell_circuit}");

    let bell_state = bell_circuit.execute().unwrap();
    println!("Bell probabilities: {:?}", bell_state.probabilities());

    // Entanglement
    let amps: Vec<(f64, f64)> = (0..4).map(|i| bell_state.amplitude(i).unwrap()).collect();
    let c = entanglement::concurrence_pure(&amps);
    println!("Concurrence: {c:.4} (1.0 = maximally entangled)");

    // Schmidt decomposition
    let coeffs = entanglement::schmidt_decomposition(&amps, 2, 2).unwrap();
    println!("Schmidt coefficients: {coeffs:?}");

    // --- Measurement ---
    let (bit, collapsed) = bell_state.measure_qubit(0, 0.3).unwrap();
    println!("\nMeasured qubit 0: {bit}");
    println!("Collapsed state: {:?}", collapsed.probabilities());

    // --- Quantum algorithms ---
    // QFT
    let qft = Circuit::qft(3);
    let qft_state = qft.execute().unwrap();
    println!("\nQFT|000⟩ probabilities: {:?}", qft_state.probabilities());

    // Deutsch-Jozsa (balanced oracle)
    let dj = Circuit::deutsch_jozsa(2, |circuit, inputs, output| {
        circuit.cnot(inputs[0], output).unwrap();
    });
    let (_state, results) = dj.execute_with_measurement(&[0.5, 0.5]).unwrap();
    let balanced = results.iter().any(|&(_, b)| b == 1);
    println!(
        "Deutsch-Jozsa: oracle is {}",
        if balanced { "balanced" } else { "constant" }
    );

    // --- Noise ---
    let dm = entanglement::DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
    let noisy = entanglement::NoiseChannel::depolarizing(0.3)
        .unwrap()
        .apply(&dm)
        .unwrap();
    println!("\nPure state purity:  {:.4}", dm.purity());
    println!("After 30% depol:   {:.4}", noisy.purity());
}
