//! Integration tests for kana.

use kana::state::StateVector;
use kana::operator::Operator;
use kana::entanglement::{self, DensityMatrix};
use kana::circuit::Circuit;

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
    let combined = zero.tensor_product(&one);

    let x = Operator::pauli_x();
    let id = Operator::identity(2);
    let x_on_first = x.tensor_product(&id);

    let result = x_on_first.apply(&combined).unwrap();
    // X⊗I |01⟩ = |11⟩ → index 3
    assert!((result.probability(3).unwrap() - 1.0).abs() < 1e-10);
}
