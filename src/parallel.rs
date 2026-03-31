//! Parallel quantum operations via rayon.
//!
//! Provides parallel versions of statevector gate application for systems
//! with many qubits (≥ 10). Below that threshold, the sequential version
//! is faster due to lower overhead.

use rayon::prelude::*;

use crate::operator::Operator;
use crate::state::StateVector;

/// Minimum dimension (2^n) before parallelism kicks in.
/// Below this, sequential is faster.
const PAR_THRESHOLD: usize = 1024; // 10 qubits

/// Apply a 2×2 gate to a target qubit in parallel.
///
/// Same semantics as `Circuit::apply_single_qubit_direct` but splits
/// the work across threads for large statevectors.
pub fn apply_single_qubit_par(state: &mut StateVector, gate: &Operator, target: usize) {
    let n = state.num_qubits();
    let dim = state.dimension();
    let elems = gate.elements();
    let (u00_re, u00_im) = elems[0];
    let (u01_re, u01_im) = elems[1];
    let (u10_re, u10_im) = elems[2];
    let (u11_re, u11_im) = elems[3];

    let bit = 1 << (n - 1 - target);
    let amps = state.amplitudes_mut();

    if dim < PAR_THRESHOLD {
        // Sequential fallback
        for i in 0..dim {
            if i & bit != 0 {
                continue;
            }
            let j = i | bit;
            let (a_re, a_im) = amps[i];
            let (b_re, b_im) = amps[j];
            amps[i] = (
                u00_re * a_re - u00_im * a_im + u01_re * b_re - u01_im * b_im,
                u00_re * a_im + u00_im * a_re + u01_re * b_im + u01_im * b_re,
            );
            amps[j] = (
                u10_re * a_re - u10_im * a_im + u11_re * b_re - u11_im * b_im,
                u10_re * a_im + u10_im * a_re + u11_re * b_im + u11_im * b_re,
            );
        }
        return;
    }

    // Collect pair indices where target bit = 0
    let indices: Vec<usize> = (0..dim).filter(|&i| i & bit == 0).collect();

    // Process pairs in parallel, writing results to a buffer
    let results: Vec<((f64, f64), (f64, f64))> = indices
        .par_iter()
        .map(|&i| {
            let j = i | bit;
            let (a_re, a_im) = amps[i];
            let (b_re, b_im) = amps[j];
            let new_i = (
                u00_re * a_re - u00_im * a_im + u01_re * b_re - u01_im * b_im,
                u00_re * a_im + u00_im * a_re + u01_re * b_im + u01_im * b_re,
            );
            let new_j = (
                u10_re * a_re - u10_im * a_im + u11_re * b_re - u11_im * b_im,
                u10_re * a_im + u10_im * a_re + u11_re * b_im + u11_im * b_re,
            );
            (new_i, new_j)
        })
        .collect();

    // Write back
    for (&i, (new_i, new_j)) in indices.iter().zip(results.iter()) {
        amps[i] = *new_i;
        amps[i | bit] = *new_j;
    }
}

/// Apply a 4×4 gate to two target qubits in parallel.
pub fn apply_two_qubit_par(state: &mut StateVector, gate: &Operator, q0: usize, q1: usize) {
    let n = state.num_qubits();
    let dim = state.dimension();
    let elems = gate.elements();
    let bit0 = 1 << (n - 1 - q0);
    let bit1 = 1 << (n - 1 - q1);
    let mask = bit0 | bit1;
    let amps = state.amplitudes_mut();

    if dim < PAR_THRESHOLD {
        // Sequential fallback
        for i in 0..dim {
            if i & mask != 0 {
                continue;
            }
            let indices = [i, i | bit1, i | bit0, i | bit0 | bit1];
            let a = [
                amps[indices[0]],
                amps[indices[1]],
                amps[indices[2]],
                amps[indices[3]],
            ];
            for (out_idx, &target_i) in indices.iter().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (in_idx, &(s_re, s_im)) in a.iter().enumerate() {
                    let (m_re, m_im) = elems[out_idx * 4 + in_idx];
                    re += m_re * s_re - m_im * s_im;
                    im += m_re * s_im + m_im * s_re;
                }
                amps[target_i] = (re, im);
            }
        }
        return;
    }

    let group_indices: Vec<usize> = (0..dim).filter(|&i| i & mask == 0).collect();

    let results: Vec<[(f64, f64); 4]> = group_indices
        .par_iter()
        .map(|&i| {
            let indices = [i, i | bit1, i | bit0, i | bit0 | bit1];
            let a = [
                amps[indices[0]],
                amps[indices[1]],
                amps[indices[2]],
                amps[indices[3]],
            ];
            let mut out = [(0.0, 0.0); 4];
            for (out_idx, slot) in out.iter_mut().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (in_idx, &(s_re, s_im)) in a.iter().enumerate() {
                    let (m_re, m_im) = elems[out_idx * 4 + in_idx];
                    re += m_re * s_re - m_im * s_im;
                    im += m_re * s_im + m_im * s_re;
                }
                *slot = (re, im);
            }
            out
        })
        .collect();

    for (&i, result) in group_indices.iter().zip(results.iter()) {
        let indices = [i, i | bit1, i | bit0, i | bit0 | bit1];
        for (k, &target_i) in indices.iter().enumerate() {
            amps[target_i] = result[k];
        }
    }
}

/// Apply a 8×8 gate to three target qubits in parallel.
pub fn apply_three_qubit_par(
    state: &mut StateVector,
    gate: &Operator,
    q0: usize,
    q1: usize,
    q2: usize,
) {
    let n = state.num_qubits();
    let dim = state.dimension();
    let elems = gate.elements();
    let bit0 = 1 << (n - 1 - q0);
    let bit1 = 1 << (n - 1 - q1);
    let bit2 = 1 << (n - 1 - q2);
    let mask = bit0 | bit1 | bit2;
    let amps = state.amplitudes_mut();

    if dim < PAR_THRESHOLD {
        for i in 0..dim {
            if i & mask != 0 {
                continue;
            }
            let indices = [
                i,
                i | bit2,
                i | bit1,
                i | bit1 | bit2,
                i | bit0,
                i | bit0 | bit2,
                i | bit0 | bit1,
                i | bit0 | bit1 | bit2,
            ];
            let a: [(f64, f64); 8] = std::array::from_fn(|k| amps[indices[k]]);
            for (out_idx, &target_i) in indices.iter().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (in_idx, &(s_re, s_im)) in a.iter().enumerate() {
                    let (m_re, m_im) = elems[out_idx * 8 + in_idx];
                    re += m_re * s_re - m_im * s_im;
                    im += m_re * s_im + m_im * s_re;
                }
                amps[target_i] = (re, im);
            }
        }
        return;
    }

    let group_indices: Vec<usize> = (0..dim).filter(|&i| i & mask == 0).collect();

    let results: Vec<[(f64, f64); 8]> = group_indices
        .par_iter()
        .map(|&i| {
            let indices = [
                i,
                i | bit2,
                i | bit1,
                i | bit1 | bit2,
                i | bit0,
                i | bit0 | bit2,
                i | bit0 | bit1,
                i | bit0 | bit1 | bit2,
            ];
            let a: [(f64, f64); 8] = std::array::from_fn(|k| amps[indices[k]]);
            let mut out = [(0.0, 0.0); 8];
            for (out_idx, slot) in out.iter_mut().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (in_idx, &(s_re, s_im)) in a.iter().enumerate() {
                    let (m_re, m_im) = elems[out_idx * 8 + in_idx];
                    re += m_re * s_re - m_im * s_im;
                    im += m_re * s_im + m_im * s_re;
                }
                *slot = (re, im);
            }
            out
        })
        .collect();

    for (&i, result) in group_indices.iter().zip(results.iter()) {
        let indices = [
            i,
            i | bit2,
            i | bit1,
            i | bit1 | bit2,
            i | bit0,
            i | bit0 | bit2,
            i | bit0 | bit1,
            i | bit0 | bit1 | bit2,
        ];
        for (k, &target_i) in indices.iter().enumerate() {
            amps[target_i] = result[k];
        }
    }
}

/// Parallel Born-rule sampling: generate multiple measurement outcomes.
///
/// Each `r` value must be in \[0, 1). Returns an error for invalid values.
pub fn sample_par(state: &StateVector, random_values: &[f64]) -> crate::error::Result<Vec<usize>> {
    for &r in random_values {
        if !(0.0..1.0).contains(&r) {
            return Err(crate::error::KanaError::InvalidParameter {
                reason: format!("random value {r} not in [0, 1)"),
            });
        }
    }
    let probs = state.probabilities();
    Ok(random_values
        .par_iter()
        .map(|&r| {
            let mut cumulative = 0.0;
            let mut outcome = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    outcome = i;
                    break;
                }
            }
            outcome
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_single_qubit_matches_sequential() {
        let h = Operator::hadamard();
        let mut state_seq = StateVector::zero(1);
        let mut state_par = StateVector::zero(1);

        // Apply sequentially (via circuit internals — replicate here)
        crate::circuit::Circuit::apply_single_qubit_direct(&mut state_seq, &h, 0);
        apply_single_qubit_par(&mut state_par, &h, 0);

        for i in 0..2 {
            let (a_re, a_im) = state_seq.amplitude(i).unwrap();
            let (b_re, b_im) = state_par.amplitude(i).unwrap();
            assert!((a_re - b_re).abs() < 1e-10);
            assert!((a_im - b_im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_par_two_qubit_matches_sequential() {
        let cnot = Operator::cnot();
        // Start with |+0⟩ = (|00⟩ + |10⟩)/√2
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut state_seq =
            StateVector::new(vec![(s, 0.0), (0.0, 0.0), (s, 0.0), (0.0, 0.0)]).unwrap();
        let mut state_par = state_seq.clone();

        crate::circuit::Circuit::apply_two_qubit_direct(&mut state_seq, &cnot, 0, 1);
        apply_two_qubit_par(&mut state_par, &cnot, 0, 1);

        for i in 0..4 {
            let (a_re, a_im) = state_seq.amplitude(i).unwrap();
            let (b_re, b_im) = state_par.amplitude(i).unwrap();
            assert!((a_re - b_re).abs() < 1e-10);
            assert!((a_im - b_im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_par_sample() {
        let state = StateVector::plus();
        let rs: Vec<f64> = (0..10).map(|i| (i as f64) / 10.0).collect();
        let par_results = sample_par(&state, &rs).unwrap();
        let seq_results = state.sample(&rs).unwrap();
        assert_eq!(par_results, seq_results);
    }
}
