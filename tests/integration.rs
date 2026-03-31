//! Integration tests for kana.

use kana::circuit::Circuit;
use kana::entanglement::{self, DensityMatrix};
use kana::operator::Operator;
use kana::state::StateVector;

#[test]
fn test_bell_state_via_circuit_components() {
    // Build |Φ+⟩ manually: H on qubit 0, then verify superposition
    let mut c = Circuit::new(1);
    c.hadamard(0).unwrap();
    let superposition = c.execute().unwrap();
    let p0 = superposition.probability(0).unwrap();
    let p1 = superposition.probability(1).unwrap();
    assert!((p0 - 0.5).abs() < 1e-10);
    assert!((p1 - 0.5).abs() < 1e-10);
}

#[test]
fn test_operator_preserves_norm() {
    let h = Operator::hadamard();
    let state = StateVector::zero(1);
    let result = h.apply(&state).unwrap();
    assert!((result.norm() - 1.0).abs() < 1e-10);
}

#[test]
fn test_entanglement_detection() {
    // Bell state is maximally entangled
    let bell = entanglement::bell_phi_plus();
    let c = entanglement::concurrence_pure(&bell);
    assert!((c - 1.0).abs() < 1e-10);

    // Product state is not entangled
    let product = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let c2 = entanglement::concurrence_pure(&product);
    assert!(c2.abs() < 1e-10);
}

#[test]
fn test_density_matrix_from_state_vector() {
    let state = StateVector::plus();
    let amps = vec![state.amplitude(0).unwrap(), state.amplitude(1).unwrap()];
    let dm = DensityMatrix::from_pure_state(&amps);
    assert!((dm.purity() - 1.0).abs() < 1e-10);
}

#[test]
fn test_reduced_density_matrix_entropy() {
    let bell = entanglement::bell_phi_plus();
    let dm = DensityMatrix::from_pure_state(&bell);
    let reduced = dm.partial_trace_b(2, 2).unwrap();
    // Maximally mixed → entropy = 1 bit
    assert!((reduced.von_neumann_entropy() - 1.0).abs() < 1e-5);
}

#[test]
fn test_circuit_execution_matches_manual() {
    // Circuit H|0⟩ should equal StateVector::plus()
    let mut c = Circuit::new(1);
    c.hadamard(0).unwrap();
    let circuit_result = c.execute().unwrap();
    let manual_result = StateVector::plus();

    let (re, im) = circuit_result.inner_product(&manual_result).unwrap();
    assert!((re - 1.0).abs() < 1e-10);
    assert!(im.abs() < 1e-10);
}

#[test]
fn test_pauli_algebra() {
    // XY = iZ
    let x = Operator::pauli_x();
    let y = Operator::pauli_y();
    let xy = x.multiply(&y).unwrap();
    let z = Operator::pauli_z();
    // xy should be i*Z
    for i in 0..2 {
        for j in 0..2 {
            let (xy_re, xy_im) = xy.element(i, j).unwrap();
            let (z_re, z_im) = z.element(i, j).unwrap();
            // i*Z: (re, im) -> (-im, re)
            assert!((xy_re - (-z_im)).abs() < 1e-10);
            assert!((xy_im - z_re).abs() < 1e-10);
        }
    }
}

#[test]
fn test_tensor_product_state_and_operator() {
    let zero = StateVector::zero(1);
    let one = StateVector::one();
    let combined = zero.tensor_product(&one).unwrap();

    let x = Operator::pauli_x();
    let id = Operator::identity(2);
    let x_on_first = x.tensor_product(&id);

    let result = x_on_first.apply(&combined).unwrap();
    // X⊗I |01⟩ = |11⟩ → index 3
    assert!((result.probability(3).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_bell_state_via_cnot_circuit() {
    // H on qubit 0, CNOT(0,1) → |Φ+⟩ = (|00⟩ + |11⟩)/√2
    let mut c = Circuit::new(2);
    c.hadamard(0).unwrap();
    c.cnot(0, 1).unwrap();
    let result = c.execute().unwrap();

    // Extract amplitudes and verify against analytical Bell state
    let bell = entanglement::bell_phi_plus();
    for (i, &(b_re, b_im)) in bell.iter().enumerate() {
        let (r_re, r_im) = result.amplitude(i).unwrap();
        assert!((r_re - b_re).abs() < 1e-10);
        assert!((r_im - b_im).abs() < 1e-10);
    }

    // Verify entanglement via concurrence
    let amps: Vec<(f64, f64)> = (0..4).map(|i| result.amplitude(i).unwrap()).collect();
    let c_val = entanglement::concurrence_pure(&amps);
    assert!((c_val - 1.0).abs() < 1e-10);
}

#[test]
fn test_ghz_state_3q() {
    // GHZ state: H(0), CNOT(0,1), CNOT(0,2) → (|000⟩ + |111⟩)/√2
    let mut c = Circuit::new(3);
    c.hadamard(0).unwrap();
    c.cnot(0, 1).unwrap();
    c.cnot(0, 2).unwrap();
    let result = c.execute().unwrap();
    let p000 = result.probability(0).unwrap(); // |000⟩
    let p111 = result.probability(7).unwrap(); // |111⟩
    assert!((p000 - 0.5).abs() < 1e-10);
    assert!((p111 - 0.5).abs() < 1e-10);
    // All other states should be zero
    for i in 1..7 {
        assert!(result.probability(i).unwrap().abs() < 1e-10);
    }
}

#[test]
fn test_swap_preserves_state() {
    // |01⟩ → SWAP → |10⟩ → SWAP → |01⟩
    let mut c = Circuit::new(2);
    c.pauli_x(1).unwrap(); // |00⟩ → |01⟩
    c.swap(0, 1).unwrap();
    c.swap(0, 1).unwrap();
    let result = c.execute().unwrap();
    // Back to |01⟩
    assert!((result.probability(1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_cnot_non_adjacent_equals_manual() {
    // CNOT(0,2) on |100⟩ should give |101⟩
    // Build |100⟩ manually
    let mut amps = vec![(0.0, 0.0); 8];
    amps[4] = (1.0, 0.0); // |100⟩ = index 4
    let state = StateVector::new(amps).unwrap();

    let mut c = Circuit::new(3);
    c.cnot(0, 2).unwrap();
    let result = c.execute_on(state).unwrap();
    // |100⟩ → |101⟩ = index 5
    assert!((result.probability(5).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_superdense_coding_all_messages() {
    // Superdense coding: encode 2 classical bits, decode both correctly
    for (b0, b1) in [(false, false), (false, true), (true, false), (true, true)] {
        let c = Circuit::superdense_coding(b0, b1);
        let (_state, results) = c.execute_with_measurement(&[0.5, 0.5]).unwrap();
        assert_eq!(results.len(), 2);
        let decoded_b0 = results[0].1; // qubit 0 measurement
        let decoded_b1 = results[1].1; // qubit 1 measurement
        assert_eq!(decoded_b0, b0 as usize, "b0 mismatch for ({b0}, {b1})");
        assert_eq!(decoded_b1, b1 as usize, "b1 mismatch for ({b0}, {b1})");
    }
}

#[test]
fn test_teleportation_circuit_structure() {
    let c = Circuit::teleportation();
    assert_eq!(c.num_qubits(), 3);
    // Should have: H, CNOT, CNOT, H, M, M = 6 gates
    assert_eq!(c.num_gates(), 6);
}

#[test]
fn test_teleportation_zero_state() {
    // Teleport |0⟩ (already prepared in q0 as initial state)
    let c = Circuit::teleportation();
    // Execute with measurement: both outcomes 0 → no correction needed
    let (_state, results) = c.execute_with_measurement(&[0.3, 0.3]).unwrap();
    let (m0, m1) = (results[0].1, results[1].1);
    // Re-execute the full corrected circuit
    let mut full = Circuit::teleportation();
    full.teleportation_correction(m0, m1).unwrap();
    let (final_state, _) = full.execute_with_measurement(&[0.3, 0.3]).unwrap();
    // Bob's qubit (q2) should hold |0⟩
    // In the teleported state, q2's marginal probability for 0 should be 1
    // Check: sum probabilities where qubit 2 = 0
    let mut prob_q2_zero = 0.0;
    for i in 0..8 {
        if i & 1 == 0 {
            // qubit 2 (least significant) is 0
            prob_q2_zero += final_state.probability(i).unwrap();
        }
    }
    assert!((prob_q2_zero - 1.0).abs() < 1e-10);
}

#[test]
fn test_bloch_roundtrip() {
    // Create |+⟩ state, get Bloch vector, verify it's on +x axis
    let plus = StateVector::plus();
    let (x, y, z) = plus.bloch_vector().unwrap();
    assert!((x - 1.0).abs() < 1e-10);
    assert!(y.abs() < 1e-10);
    assert!(z.abs() < 1e-10);

    // Reconstruct density matrix from Bloch vector
    let dm = entanglement::tomography_single_qubit(x, y, z);
    assert!((dm.purity() - 1.0).abs() < 1e-10);
}

#[test]
fn test_deutsch_jozsa_constant_oracle() {
    // Constant f(x) = 0 → all inputs measure 0
    let c = Circuit::deutsch_jozsa(2, |_circuit, _inputs, _output| {});
    let (_state, results) = c.execute_with_measurement(&[0.5, 0.5]).unwrap();
    assert!(results.iter().all(|&(_, bit)| bit == 0));
}

#[test]
fn test_deutsch_jozsa_balanced_oracle() {
    // Balanced f(x) = x₀ → at least one input measures 1
    let c = Circuit::deutsch_jozsa(2, |circuit, inputs, output| {
        circuit.cnot(inputs[0], output).unwrap();
    });
    let (_state, results) = c.execute_with_measurement(&[0.5, 0.5]).unwrap();
    assert!(results.iter().any(|&(_, bit)| bit == 1));
}

#[test]
fn test_qft_inverse_qft_roundtrip() {
    // QFT then inverse QFT should return to |00⟩
    let qft = Circuit::qft(2);
    let state = qft.execute().unwrap();
    let iqft = Circuit::inverse_qft(2);
    let result = iqft.execute_on(state).unwrap();
    assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_qft_3q_roundtrip() {
    let qft = Circuit::qft(3);
    let state = qft.execute().unwrap();
    let iqft = Circuit::inverse_qft(3);
    let result = iqft.execute_on(state).unwrap();
    assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_grover_2q_finds_target() {
    // 2-qubit Grover searching for |11⟩, 1 iteration (optimal for N=4)
    let c = Circuit::grover(2, 1, |circuit, qubits| {
        // Oracle: flip phase of |11⟩ via CZ
        circuit.cz(qubits[0], qubits[1]).unwrap();
    });
    let (_state, results) = c.execute_with_measurement(&[0.5, 0.5]).unwrap();
    // Should find |11⟩ with high probability
    assert_eq!(results[0].1, 1);
    assert_eq!(results[1].1, 1);
}

#[test]
fn test_vqe_ansatz_executes() {
    // 2-qubit, 1-layer VQE with zero params → should produce |00⟩
    let params = vec![0.0; 4];
    let c = Circuit::vqe_ansatz(2, 1, &params).unwrap();
    let result = c.execute().unwrap();
    assert!((result.probability(0).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_rotation_gates_unitarity() {
    use std::f64::consts::PI;
    // Rx(θ)†Rx(θ) = I
    let rx = Operator::rx(PI / 3.0);
    let rxd = rx.dagger();
    let product = rxd.multiply(&rx).unwrap();
    let id = Operator::identity(2);
    for i in 0..2 {
        for j in 0..2 {
            let (p_re, p_im) = product.element(i, j).unwrap();
            let (i_re, i_im) = id.element(i, j).unwrap();
            assert!((p_re - i_re).abs() < 1e-10);
            assert!((p_im - i_im).abs() < 1e-10);
        }
    }
}

#[test]
fn test_noise_channel_on_circuit_output() {
    // Run a circuit, convert to density matrix, apply noise
    let mut c = Circuit::new(1);
    c.hadamard(0).unwrap();
    let state = c.execute().unwrap();
    let amps: Vec<(f64, f64)> = (0..2).map(|i| state.amplitude(i).unwrap()).collect();
    let dm = entanglement::DensityMatrix::from_pure_state(&amps);

    let noisy = entanglement::NoiseChannel::depolarizing(0.5)
        .unwrap()
        .apply(&dm)
        .unwrap();
    // Noisy state should have lower purity
    assert!(noisy.purity() < dm.purity());
    // But trace should still be 1
    let (tr_re, tr_im) = noisy.trace();
    assert!((tr_re - 1.0).abs() < 1e-10);
    assert!(tr_im.abs() < 1e-10);
}
