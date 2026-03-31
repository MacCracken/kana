//! Ownership-based circuit construction with compile-time no-cloning.
//!
//! Demonstrates the safe builder API where qubits are move-only types.

use kana::safe::QuantumBuilder;

fn main() {
    // GHZ state: (|000> + |111>)/sqrt(2)
    let (mut b, [q0, q1, q2]) = QuantumBuilder::new::<3>();
    let q0 = b.h(q0);
    let (q0, q1) = b.cx(q0, q1);
    let (q0, q2) = b.cx(q0, q2);
    let circuit = b.build([q0, q1, q2]);

    println!("GHZ circuit:\n{circuit}");

    let result = circuit.execute().unwrap();
    println!("Probabilities: {:?}", result.probabilities());
    println!("|000> = {:.4}", result.probability(0).unwrap());
    println!("|111> = {:.4}", result.probability(7).unwrap());

    // Measurement
    let (mut b, [q0, q1]) = QuantumBuilder::new::<2>();
    let q0 = b.h(q0);
    let (q0, q1) = b.cx(q0, q1);
    let (bit0, q0) = b.measure(q0);
    let (bit1, q1) = b.measure(q1);
    let circuit = b.build([q0, q1]);

    println!("\nBell + measure circuit:\n{circuit}");
    println!("Measured qubits: {}, {}", bit0.qubit(), bit1.qubit());
}
