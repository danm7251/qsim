// Only compiles if the "bench" feature is enabled since otherwise many Qsim functions are private.
#![cfg(feature = "bench")]

// This benchmark will often use depracated functions for comparison.
#![allow(deprecated)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use qsim::{gates::Gate, state::State};

// For the first benchmark test I will compare the index and kron methods
fn bench_kernels_1q_over_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Idx & Kron Kernels (T=[0..=7], n=8)");
    group.measurement_time(Duration::from_secs(8));

    let n = 8;
    let targets = [0, 1, 2, 3, 4, 5, 6, 7];

    for target in targets {
        let gate = Gate::H { target };

        let mut state = State::zero(n).unwrap();
        group.bench_with_input(
            BenchmarkId::new("1Q-Idx", target),
            &target,
            |b, target| b.iter(
                || state.apply_1q_index(*target, gate.matrix())
            )
        );

        let mut state = State::zero(n).unwrap();
        group.bench_with_input(
            BenchmarkId::new("1Q-Kron", target),
            &target,
            |b, target| b.iter(
                || state.apply_1q_kron(*target, gate.matrix())
            )
        );
    }

    group.finish();
}

/// This benchmark tests the 1 Qubit Index kernel at a circuit size of 7 by applying a Hadamard gate with a target qubit
/// of the first, middle 
fn bench_1q_index_kernel_over_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Idx Kernel (t=[0, 3, 6], n=7)");

    let n = 7;
    for target in [0, 3, 6] {
        let gate = Gate::H { target };

        let mut state = State::zero(n).unwrap();
        group.bench_with_input(
            BenchmarkId::new("1Q-Index", target),
            &target,
            |b, target| b.iter(
                || state.apply_1q_index(*target, gate.matrix())
            )
        );
    }

    group.finish();
}

criterion_group!(
    // Name of function group for benchmarking.
    benches,

    // Targets to add to the function group.
    bench_kernels_1q_over_target,
    bench_1q_index_kernel_over_target
);

// Takes the function group and expands into the program's entry point.
criterion_main!(benches);