//! Noise channel simulation — depolarizing, amplitude damping, phase damping.
//!
//! Demonstrates how noise degrades quantum states.

use kana::entanglement::{self, DensityMatrix, NoiseChannel};

fn main() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let plus = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]);

    println!("Initial |+> state:");
    println!("  Purity:  {:.4}", plus.purity());
    println!("  Entropy: {:.4}", plus.von_neumann_entropy());

    // Depolarizing channel at various strengths
    println!("\nDepolarizing channel:");
    for p in [0.1, 0.3, 0.5, 1.0] {
        let ch = NoiseChannel::depolarizing(p).unwrap();
        let noisy = ch.apply(&plus).unwrap();
        println!(
            "  p={p:.1}: purity={:.4}, entropy={:.4}",
            noisy.purity(),
            noisy.von_neumann_entropy()
        );
    }

    // Amplitude damping (T1 decay)
    println!("\nAmplitude damping (|1> decay):");
    let one = DensityMatrix::from_pure_state(&[(0.0, 0.0), (1.0, 0.0)]);
    for gamma in [0.1, 0.5, 0.9, 1.0] {
        let ch = NoiseChannel::amplitude_damping(gamma).unwrap();
        let decayed = ch.apply(&one).unwrap();
        let (p0, _) = decayed.element(0, 0).unwrap();
        println!("  gamma={gamma:.1}: P(|0>)={p0:.4}");
    }

    // Phase damping (T2 dephasing)
    println!("\nPhase damping (coherence decay):");
    for gamma in [0.1, 0.5, 0.9, 1.0] {
        let ch = NoiseChannel::phase_damping(gamma).unwrap();
        let dephased = ch.apply(&plus).unwrap();
        let (re01, _) = dephased.element(0, 1).unwrap();
        println!("  gamma={gamma:.1}: Re(rho_01)={re01:.4}");
    }

    // Channel composition
    println!("\nComposed channel (depol + damping):");
    let ch1 = NoiseChannel::depolarizing(0.1).unwrap();
    let ch2 = NoiseChannel::amplitude_damping(0.1).unwrap();
    let composed = ch1.compose(&ch2).unwrap();
    let noisy = composed.apply(&plus).unwrap();
    println!("  Purity: {:.4}", noisy.purity());

    // Process fidelity
    println!("\nProcess fidelity vs identity:");
    let ch = NoiseChannel::depolarizing(0.1).unwrap();
    println!("  F_pro = {:.4}", ch.process_fidelity());
    println!("  F_avg = {:.4}", ch.average_gate_fidelity());

    // Entanglement under noise
    println!("\nBell state under depolarizing:");
    let bell = entanglement::bell_phi_plus();
    let bell_dm = DensityMatrix::from_pure_state(&bell);
    let c = entanglement::concurrence_pure(&bell);
    println!("  Concurrence (pure): {c:.4}");
    let neg = entanglement::negativity(&bell_dm, 2, 2).unwrap();
    println!("  Negativity: {neg:.4}");
}
