//! Quantum algorithms — Deutsch-Jozsa, Grover, QFT.

use kana::circuit::Circuit;

fn main() {
    // Deutsch-Jozsa: distinguish constant from balanced oracles
    println!("=== Deutsch-Jozsa ===");

    let constant = Circuit::deutsch_jozsa(2, |_c, _inputs, _output| {
        // f(x) = 0 for all x — no gates
    });
    let (_, results) = constant.execute_with_measurement(&[0.5, 0.5]).unwrap();
    let is_constant = results.iter().all(|&(_, b)| b == 0);
    println!(
        "Constant oracle: {}",
        if is_constant { "CONSTANT" } else { "BALANCED" }
    );

    let balanced = Circuit::deutsch_jozsa(2, |c, inputs, output| {
        c.cnot(inputs[0], output).unwrap(); // f(x) = x_0
    });
    let (_, results) = balanced.execute_with_measurement(&[0.5, 0.5]).unwrap();
    let is_constant = results.iter().all(|&(_, b)| b == 0);
    println!(
        "Balanced oracle: {}",
        if is_constant { "CONSTANT" } else { "BALANCED" }
    );

    // QFT roundtrip
    println!("\n=== QFT ===");
    let qft = Circuit::qft(3);
    let state = qft.execute().unwrap();
    println!(
        "QFT|000>: uniform? {:.4} per state",
        state.probability(0).unwrap()
    );

    let iqft = Circuit::inverse_qft(3);
    let back = iqft.execute_on(state).unwrap();
    println!(
        "iQFT(QFT|000>): P(|000>) = {:.4}",
        back.probability(0).unwrap()
    );

    // Grover: search for |11> in 2-qubit space
    println!("\n=== Grover (2-qubit, target |11>) ===");
    let grover = Circuit::grover(2, 1, |c, qubits| {
        c.cz(qubits[0], qubits[1]).unwrap(); // oracle marks |11>
    });
    let (_state, results) = grover.execute_with_measurement(&[0.5, 0.5]).unwrap();
    println!("Found: |{}{}>", results[0].1, results[1].1);

    // VQE ansatz
    println!("\n=== VQE Ansatz ===");
    let params = vec![0.5, 0.3, 0.7, 0.1]; // 2 qubits, 1 layer
    let vqe = Circuit::vqe_ansatz(2, 1, &params).unwrap();
    let state = vqe.execute().unwrap();
    let (idx, prob) = state.most_probable();
    println!("Most probable state: |{idx:02b}> with P={prob:.4}");
    println!("Support size: {} states", state.support_size());
}
