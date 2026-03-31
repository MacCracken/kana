//! Time evolution — Schrodinger equation and Lindblad master equation.

use kana::dynamics::{Hamiltonian, expectation_value};
use kana::entanglement::DensityMatrix;
use kana::operator::Operator;
use kana::state::StateVector;

fn main() {
    // Closed system: spin precession under Pauli-Z
    println!("=== Schrodinger Evolution ===");
    let h = Hamiltonian::new(Operator::pauli_z());
    let state = StateVector::plus();
    println!(
        "Initial: P(|0>)={:.4}, P(|1>)={:.4}",
        state.probability(0).unwrap(),
        state.probability(1).unwrap()
    );

    for t in [0.5, 1.0, std::f64::consts::PI] {
        let evolved = h.evolve_state(&state, t).unwrap();
        let (x, y, z) = evolved.bloch_vector().unwrap();
        println!("t={t:.2}: Bloch=({x:.3}, {y:.3}, {z:.3})");
    }

    // Evolution operator unitarity check
    let u = h.evolution_operator(1.0).unwrap();
    let udu = u.dagger().multiply(&u).unwrap();
    let (re, _) = udu.element(0, 0).unwrap();
    println!("U(1)^dag U(1)[0,0] = {re:.6} (should be 1.0)");

    // Open system: dephasing under Lindblad
    println!("\n=== Lindblad Dephasing ===");
    let mut h_open = Hamiltonian::new(Operator::new(2, vec![(0.0, 0.0); 4]).unwrap());
    h_open.add_dissipator(0.5, Operator::pauli_z());

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rho = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]);

    for t in [1.0, 5.0, 10.0, 20.0] {
        let evolved = h_open
            .evolve_density(&rho, t, (t * 100.0) as usize)
            .unwrap();
        let (re01, _) = evolved.element(0, 1).unwrap();
        println!(
            "t={t:5.1}: coherence |rho_01| = {:.4}, purity = {:.4}",
            re01.abs(),
            evolved.purity()
        );
    }

    // Expectation values
    println!("\n=== Expectation Values ===");
    let rho = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
    let (z_re, _) = expectation_value(&Operator::pauli_z(), &rho).unwrap();
    let (x_re, _) = expectation_value(&Operator::pauli_x(), &rho).unwrap();
    println!("|0>: <Z>={z_re:.4}, <X>={x_re:.4}");

    let rho_plus = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]);
    let (z_re, _) = expectation_value(&Operator::pauli_z(), &rho_plus).unwrap();
    let (x_re, _) = expectation_value(&Operator::pauli_x(), &rho_plus).unwrap();
    println!("|+>: <Z>={z_re:.4}, <X>={x_re:.4}");
}
