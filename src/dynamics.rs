//! Quantum dynamics — time evolution for closed and open quantum systems.
//!
//! - **Schrödinger equation**: |ψ(t)⟩ = e^(−iHt) |ψ(0)⟩ for closed systems
//! - **Lindblad master equation**: dρ/dt = −i[H,ρ] + Σₖ γₖ D[Lₖ](ρ) for open systems
//!   where D[L](ρ) = LρL† − ½{L†L, ρ}

use crate::entanglement::DensityMatrix;
use crate::error::{KanaError, Result};
use crate::operator::Operator;
use crate::state::StateVector;

/// A Hamiltonian with optional Lindblad dissipators for open system dynamics.
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    /// The Hermitian Hamiltonian operator H.
    operator: Operator,
    /// Lindblad dissipators: (rate γₖ, collapse operator Lₖ).
    dissipators: Vec<(f64, Operator)>,
}

impl Hamiltonian {
    /// Create a closed-system Hamiltonian (no dissipation).
    pub fn new(operator: Operator) -> Self {
        Self {
            operator,
            dissipators: Vec::new(),
        }
    }

    /// Create an open-system Hamiltonian with Lindblad dissipators.
    ///
    /// Each dissipator is a `(rate, collapse_operator)` pair.
    pub fn with_dissipators(operator: Operator, dissipators: Vec<(f64, Operator)>) -> Self {
        Self {
            operator,
            dissipators,
        }
    }

    /// Add a Lindblad dissipator with rate γ and collapse operator L.
    pub fn add_dissipator(&mut self, rate: f64, collapse_op: Operator) {
        self.dissipators.push((rate, collapse_op));
    }

    /// Check if this is an open system (has dissipators).
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.dissipators.is_empty()
    }

    /// Dimension of the Hilbert space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.operator.dim()
    }

    /// Schrödinger evolution: |ψ(t)⟩ = e^(−iHt) |ψ(0)⟩.
    ///
    /// Uses matrix exponentiation via Taylor series (Padé approximation
    /// for small dt, or eigendecomposition for larger time steps).
    ///
    /// Only valid for closed systems (no dissipators).
    pub fn evolve_state(&self, state: &StateVector, dt: f64) -> Result<StateVector> {
        if self.is_open() {
            return Err(KanaError::InvalidParameter {
                reason: "use evolve_density for open systems with dissipators".into(),
            });
        }
        if self.operator.dim() != state.dimension() {
            return Err(KanaError::DimensionMismatch {
                expected: self.operator.dim(),
                got: state.dimension(),
            });
        }
        // U(t) = e^(-iHt) computed via Taylor series: U ≈ Σ (−iHt)^n / n!
        let u = self.evolution_operator(dt)?;
        u.apply(state)
    }

    /// Compute the time-evolution operator U(t) = e^(−iHt).
    pub fn evolution_operator(&self, dt: f64) -> Result<Operator> {
        let dim = self.operator.dim();
        let h_elems = self.operator.elements();

        // Build -iHt matrix
        let mut m = vec![(0.0, 0.0); dim * dim];
        for (idx, &(h_re, h_im)) in h_elems.iter().enumerate() {
            // -i * (h_re + i*h_im) * dt = (h_im * dt) + i*(-h_re * dt)
            m[idx] = (h_im * dt, -h_re * dt);
        }

        // Matrix exponential via Taylor series: e^M ≈ I + M + M²/2! + M³/3! + ...
        let mut result = vec![(0.0, 0.0); dim * dim];
        // Start with identity
        for i in 0..dim {
            result[i * dim + i] = (1.0, 0.0);
        }

        let mut term = result.clone(); // Current term: M^n / n!
        for n in 1..30 {
            // term = term * M / n
            let mut new_term = vec![(0.0, 0.0); dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    let (mut re, mut im) = (0.0, 0.0);
                    for k in 0..dim {
                        let (t_re, t_im) = term[i * dim + k];
                        let (m_re, m_im) = m[k * dim + j];
                        re += t_re * m_re - t_im * m_im;
                        im += t_re * m_im + t_im * m_re;
                    }
                    new_term[i * dim + j] = (re / n as f64, im / n as f64);
                }
            }
            term = new_term;

            // Add term to result
            for i in 0..dim * dim {
                result[i].0 += term[i].0;
                result[i].1 += term[i].1;
            }

            // Check convergence
            let norm: f64 = term.iter().map(|(re, im)| re * re + im * im).sum();
            if norm < 1e-30 {
                break;
            }
        }

        Operator::new(dim, result)
    }

    /// Lindblad master equation evolution of a density matrix.
    ///
    /// dρ/dt = −i[H,ρ] + Σₖ γₖ (Lₖ ρ Lₖ† − ½{Lₖ†Lₖ, ρ})
    ///
    /// Uses Euler method with the given time step. For accuracy, use small dt
    /// or call multiple times.
    pub fn evolve_density(
        &self,
        rho: &DensityMatrix,
        dt: f64,
        steps: usize,
    ) -> Result<DensityMatrix> {
        let dim = self.operator.dim();
        if rho.dim() != dim {
            return Err(KanaError::DimensionMismatch {
                expected: dim,
                got: rho.dim(),
            });
        }

        let mut current = rho.elements().to_vec();
        let step_dt = dt / steps as f64;

        for _ in 0..steps {
            let deriv = self.lindblad_rhs(&current, dim);
            // Euler step: ρ(t+dt) = ρ(t) + dt · dρ/dt
            for i in 0..dim * dim {
                current[i].0 += step_dt * deriv[i].0;
                current[i].1 += step_dt * deriv[i].1;
            }
        }

        DensityMatrix::new(dim, current)
    }

    /// Compute the Lindblad right-hand side: dρ/dt = −i[H,ρ] + dissipator terms.
    fn lindblad_rhs(&self, rho: &[(f64, f64)], dim: usize) -> Vec<(f64, f64)> {
        let h = self.operator.elements();
        let mut result = vec![(0.0, 0.0); dim * dim];

        // Unitary part: −i[H,ρ] = −i(Hρ − ρH)
        for i in 0..dim {
            for j in 0..dim {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..dim {
                    // H[i][k] * ρ[k][j]
                    let (h_re, h_im) = h[i * dim + k];
                    let (r_re, r_im) = rho[k * dim + j];
                    re += h_re * r_re - h_im * r_im;
                    im += h_re * r_im + h_im * r_re;
                    // − ρ[i][k] * H[k][j]
                    let (r2_re, r2_im) = rho[i * dim + k];
                    let (h2_re, h2_im) = h[k * dim + j];
                    re -= r2_re * h2_re - r2_im * h2_im;
                    im -= r2_re * h2_im + r2_im * h2_re;
                }
                // Multiply by -i: -i*(re + i*im) = im - i*re
                result[i * dim + j].0 += im;
                result[i * dim + j].1 += -re;
            }
        }

        // Dissipator terms: Σ γₖ (Lₖ ρ Lₖ† − ½ Lₖ†Lₖ ρ − ½ ρ Lₖ†Lₖ)
        for (gamma, l_op) in &self.dissipators {
            let l = l_op.elements();
            // Precompute L†L
            let mut ldl = vec![(0.0, 0.0); dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        let (l_ki_re, l_ki_im) = l[k * dim + i];
                        let (l_kj_re, l_kj_im) = l[k * dim + j];
                        // L†[i][k] * L[k][j] = conj(L[k][i]) * L[k][j]
                        ldl[i * dim + j].0 += l_ki_re * l_kj_re + l_ki_im * l_kj_im;
                        ldl[i * dim + j].1 += l_ki_re * l_kj_im - l_ki_im * l_kj_re;
                    }
                }
            }

            for i in 0..dim {
                for j in 0..dim {
                    let (mut re, mut im) = (0.0, 0.0);
                    for k in 0..dim {
                        for m in 0..dim {
                            // L ρ L†: L[i][k] * ρ[k][m] * conj(L[j][m])
                            let (l_ik_re, l_ik_im) = l[i * dim + k];
                            let (r_km_re, r_km_im) = rho[k * dim + m];
                            let (l_jm_re, l_jm_im) = l[j * dim + m];
                            let temp_re = l_ik_re * r_km_re - l_ik_im * r_km_im;
                            let temp_im = l_ik_re * r_km_im + l_ik_im * r_km_re;
                            re += temp_re * l_jm_re + temp_im * l_jm_im;
                            im += temp_im * l_jm_re - temp_re * l_jm_im;
                        }

                        // -½ L†L ρ: -0.5 * ldl[i][k] * ρ[k][j]
                        let (ldl_re, ldl_im) = ldl[i * dim + k];
                        let (r_re, r_im) = rho[k * dim + j];
                        re -= 0.5 * (ldl_re * r_re - ldl_im * r_im);
                        im -= 0.5 * (ldl_re * r_im + ldl_im * r_re);

                        // -½ ρ L†L: -0.5 * ρ[i][k] * ldl[k][j]
                        let (r2_re, r2_im) = rho[i * dim + k];
                        let (ldl2_re, ldl2_im) = ldl[k * dim + j];
                        re -= 0.5 * (r2_re * ldl2_re - r2_im * ldl2_im);
                        im -= 0.5 * (r2_re * ldl2_im + r2_im * ldl2_re);
                    }
                    result[i * dim + j].0 += gamma * re;
                    result[i * dim + j].1 += gamma * im;
                }
            }
        }

        result
    }
}

/// Compute the expectation value ⟨O⟩ = Tr(O ρ) for an observable O and state ρ.
pub fn expectation_value(observable: &Operator, rho: &DensityMatrix) -> Result<(f64, f64)> {
    if observable.dim() != rho.dim() {
        return Err(KanaError::DimensionMismatch {
            expected: observable.dim(),
            got: rho.dim(),
        });
    }
    let dim = observable.dim();
    let o = observable.elements();
    let (mut re, mut im) = (0.0, 0.0);
    for i in 0..dim {
        for j in 0..dim {
            let (o_re, o_im) = o[i * dim + j];
            let (r_re, r_im) = rho.element(j, i).unwrap();
            re += o_re * r_re - o_im * r_im;
            im += o_re * r_im + o_im * r_re;
        }
    }
    Ok((re, im))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schrodinger_identity_hamiltonian() {
        // H = 0 → U(t) = I → state unchanged
        let h = Hamiltonian::new(Operator::new(2, vec![(0.0, 0.0); 4]).unwrap());
        let state = StateVector::zero(1);
        let evolved = h.evolve_state(&state, 1.0).unwrap();
        assert!((evolved.probability(0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_schrodinger_pauli_z() {
        // H = Z, |+⟩ evolves with phase oscillation
        let h = Hamiltonian::new(Operator::pauli_z());
        let state = StateVector::plus();
        let evolved = h.evolve_state(&state, std::f64::consts::FRAC_PI_2).unwrap();
        // At t = π/2: e^(-iZπ/2)|+⟩ should still have 50/50 probabilities
        let p0 = evolved.probability(0).unwrap();
        let p1 = evolved.probability(1).unwrap();
        assert!((p0 - 0.5).abs() < 1e-5);
        assert!((p1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_evolution_operator_unitarity() {
        let h = Hamiltonian::new(Operator::pauli_x());
        let u = h.evolution_operator(0.5).unwrap();
        // U†U should be ≈ I
        let udu = u.dagger().multiply(&u).unwrap();
        let id = Operator::identity(2);
        for i in 0..2 {
            for j in 0..2 {
                let (a_re, a_im) = udu.element(i, j).unwrap();
                let (b_re, b_im) = id.element(i, j).unwrap();
                assert!((a_re - b_re).abs() < 1e-8);
                assert!((a_im - b_im).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_lindblad_pure_dephasing() {
        // H = 0, one dephasing dissipator L = Z, γ = 0.1
        // Starting from |+⟩, off-diagonal should decay
        let mut h = Hamiltonian::new(Operator::new(2, vec![(0.0, 0.0); 4]).unwrap());
        h.add_dissipator(0.1, Operator::pauli_z());

        let s = std::f64::consts::FRAC_1_SQRT_2;
        let rho = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]);
        let evolved = h.evolve_density(&rho, 10.0, 1000).unwrap();

        // Off-diagonal should have decayed
        let (re01, _im01) = evolved.element(0, 1).unwrap();
        assert!(re01.abs() < 0.2); // should be much smaller than initial 0.5

        // Diagonal should be preserved (no energy exchange)
        let (re00, _) = evolved.element(0, 0).unwrap();
        assert!((re00 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_lindblad_amplitude_damping() {
        // H = 0, amplitude damping L = |0⟩⟨1|, γ = 0.5
        // Starting from |1⟩, should decay toward |0⟩
        let mut h = Hamiltonian::new(Operator::new(2, vec![(0.0, 0.0); 4]).unwrap());
        let l01 = Operator::new(2, vec![(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]).unwrap();
        h.add_dissipator(0.5, l01);

        let rho = DensityMatrix::from_pure_state(&[(0.0, 0.0), (1.0, 0.0)]);
        let evolved = h.evolve_density(&rho, 5.0, 500).unwrap();

        // Should be mostly |0⟩ after strong damping
        let (re00, _) = evolved.element(0, 0).unwrap();
        assert!(re00 > 0.8);
    }

    #[test]
    fn test_expectation_value_z() {
        // ⟨Z⟩ for |0⟩ should be 1
        let rho = DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);
        let (re, im) = expectation_value(&Operator::pauli_z(), &rho).unwrap();
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_expectation_value_x_plus() {
        // ⟨X⟩ for |+⟩ should be 1
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let rho = DensityMatrix::from_pure_state(&[(s, 0.0), (s, 0.0)]);
        let (re, im) = expectation_value(&Operator::pauli_x(), &rho).unwrap();
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_evolve_state_rejects_open_system() {
        let mut h = Hamiltonian::new(Operator::pauli_z());
        h.add_dissipator(0.1, Operator::pauli_z());
        let state = StateVector::zero(1);
        assert!(h.evolve_state(&state, 1.0).is_err());
    }
}
