use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn state_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("state");

    group.bench_function("zero_1q", |b| {
        b.iter(|| black_box(kana::state::StateVector::zero(1)))
    });

    group.bench_function("zero_4q", |b| {
        b.iter(|| black_box(kana::state::StateVector::zero(4)))
    });

    group.bench_function("zero_8q", |b| {
        b.iter(|| black_box(kana::state::StateVector::zero(8)))
    });

    group.bench_function("plus", |b| {
        b.iter(|| black_box(kana::state::StateVector::plus()))
    });

    let a = kana::state::StateVector::zero(1);
    let b_state = kana::state::StateVector::one();
    group.bench_function("inner_product_1q", |bench| {
        bench.iter(|| black_box(a.inner_product(&b_state)))
    });

    let z4 = kana::state::StateVector::zero(4);
    group.bench_function("probabilities_4q", |bench| {
        bench.iter(|| black_box(z4.probabilities()))
    });

    let z1 = kana::state::StateVector::zero(1);
    let o1 = kana::state::StateVector::one();
    group.bench_function("tensor_product_1q", |bench| {
        bench.iter(|| black_box(z1.tensor_product(&o1).unwrap()))
    });

    group.finish();
}

fn operator_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("operator");

    group.bench_function("hadamard_create", |b| {
        b.iter(|| black_box(kana::operator::Operator::hadamard()))
    });

    let h = kana::operator::Operator::hadamard();
    let state = kana::state::StateVector::zero(1);
    group.bench_function("hadamard_apply_1q", |b| {
        b.iter(|| black_box(h.apply(&state)))
    });

    let x = kana::operator::Operator::pauli_x();
    let y = kana::operator::Operator::pauli_y();
    group.bench_function("multiply_2x2", |b| b.iter(|| black_box(x.multiply(&y))));

    group.bench_function("dagger_2x2", |b| b.iter(|| black_box(h.dagger())));

    let id = kana::operator::Operator::identity(2);
    group.bench_function("tensor_product_2x2", |b| {
        b.iter(|| black_box(x.tensor_product(&id)))
    });

    group.bench_function("cnot_create", |b| {
        b.iter(|| black_box(kana::operator::Operator::cnot()))
    });

    let cnot = kana::operator::Operator::cnot();
    let state_2q = kana::state::StateVector::zero(2);
    group.bench_function("cnot_apply_2q", |b| {
        b.iter(|| black_box(cnot.apply(&state_2q)))
    });

    group.finish();
}

fn entanglement_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("entanglement");

    group.bench_function("bell_phi_plus", |b| {
        b.iter(|| black_box(kana::entanglement::bell_phi_plus()))
    });

    let bell = kana::entanglement::bell_phi_plus();
    group.bench_function("concurrence_pure", |b| {
        b.iter(|| black_box(kana::entanglement::concurrence_pure(&bell)))
    });

    group.bench_function("density_matrix_2q", |b| {
        b.iter(|| black_box(kana::entanglement::DensityMatrix::from_pure_state(&bell)))
    });

    let dm = kana::entanglement::DensityMatrix::from_pure_state(&bell);
    group.bench_function("partial_trace_2q", |b| {
        b.iter(|| black_box(dm.partial_trace_b(2, 2)))
    });

    group.bench_function("purity_2q", |b| b.iter(|| black_box(dm.purity())));

    let reduced = dm.partial_trace_b(2, 2).unwrap();
    group.bench_function("von_neumann_entropy_2x2", |b| {
        b.iter(|| black_box(reduced.von_neumann_entropy()))
    });

    group.bench_function("von_neumann_entropy_4x4", |b| {
        b.iter(|| black_box(dm.von_neumann_entropy()))
    });

    group.finish();
}

fn circuit_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit");

    group.bench_function("execute_1q_1gate", |b| {
        let mut c = kana::circuit::Circuit::new(1);
        c.hadamard(0).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("execute_2q_2gate", |b| {
        let mut c = kana::circuit::Circuit::new(2);
        c.hadamard(0).unwrap();
        c.pauli_x(1).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("execute_3q_3gate", |b| {
        let mut c = kana::circuit::Circuit::new(3);
        c.hadamard(0).unwrap();
        c.pauli_x(1).unwrap();
        c.pauli_z(2).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("bell_circuit_2q", |b| {
        let mut c = kana::circuit::Circuit::new(2);
        c.hadamard(0).unwrap();
        c.cnot(0, 1).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("cnot_non_adjacent_3q", |b| {
        let mut c = kana::circuit::Circuit::new(3);
        c.hadamard(0).unwrap();
        c.cnot(0, 2).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("qft_3q", |b| {
        let c = kana::circuit::Circuit::qft(3);
        b.iter(|| black_box(c.execute()))
    });

    group.bench_function("toffoli_3q", |b| {
        let mut c = kana::circuit::Circuit::new(3);
        c.toffoli(0, 1, 2).unwrap();
        b.iter(|| black_box(c.execute()))
    });

    group.finish();
}

fn noise_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise");

    let dm = kana::entanglement::DensityMatrix::from_pure_state(&[(1.0, 0.0), (0.0, 0.0)]);

    group.bench_function("depolarizing_apply", |b| {
        let ch = kana::entanglement::NoiseChannel::depolarizing(0.1).unwrap();
        b.iter(|| black_box(ch.apply(&dm)))
    });

    group.bench_function("amplitude_damping_apply", |b| {
        let ch = kana::entanglement::NoiseChannel::amplitude_damping(0.1).unwrap();
        b.iter(|| black_box(ch.apply(&dm)))
    });

    group.bench_function("phase_damping_apply", |b| {
        let ch = kana::entanglement::NoiseChannel::phase_damping(0.1).unwrap();
        b.iter(|| black_box(ch.apply(&dm)))
    });

    group.finish();
}

criterion_group!(
    benches,
    state_benchmarks,
    operator_benchmarks,
    entanglement_benchmarks,
    circuit_benchmarks,
    noise_benchmarks,
);
criterion_main!(benches);
