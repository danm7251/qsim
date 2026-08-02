// Only compiles if the "bench" feature is enabled since otherwise many Qsim functions are private.
#![cfg(feature = "bench")]

// This benchmark will often use depracated functions for comparison.
#![allow(deprecated)]

use std::{hint::black_box, time::Duration};

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

// CUSTOM VS NDARRAY BENCHES

// Test construction of Matrix class against ndarray::Array2
fn bench_matrix_construction(c: &mut Criterion) {
    use qsim::linalg::SquareMatrix;
    use ndarray::Array2;
    use num_complex::Complex;

    let mut group = c.benchmark_group("Matrix Construction");

    // Sizes of N x N matrices.
    let sizes = [2, 4, 8, 16, 32, 64, 128, 256];

    for size in sizes {

        group.bench_with_input(
            BenchmarkId::new("Matrix", size),
            &size,
            |b, &size| b.iter(
                || black_box(SquareMatrix::zero(size))
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Ndarray", size),
            &size,
            |b, &size| b.iter(
                || black_box(Array2::<Complex<f64>>::zeros((size, size)))
            )
        );
    }

    group.finish();
}

// Test construction of Vector class against ndarray::Array1
fn bench_vector_construction(c: &mut Criterion) {
    use qsim::linalg::Vector;
    use ndarray::Array1;
    use num_complex::Complex;

    let mut group = c.benchmark_group("Vector Construction");

    // Sizes of vectors.
    let sizes = [
        1,
        4,
        16,
        64,
        256,
        1_024,
        4_096,
        16_384,
        65_536,
        262_144,
        1_048_576,
    ];

    for size in sizes {

        group.bench_with_input(
            BenchmarkId::new("Vector", size),
            &size,
            |b, &size| b.iter(
                || black_box(Vector::zeros(size))
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Array1", size),
            &size,
            |b, &size| b.iter(
                || black_box(Array1::<Complex<f64>>::zeros(size))
            )
        );
    }

    group.finish();
}

// Test traversing Matrix sequentially against ndarray::Array2
fn bench_matrix_sequential_traversal(c: &mut Criterion) {
    use qsim::linalg::SquareMatrix;
    use ndarray::Array2;
    use num_complex::Complex;

    let mut group = c.benchmark_group("Matrix Sequential Read");
    group.measurement_time(Duration::from_secs(10));

    // Sizes of N x N matrices.
    let parameters = [(16, 4096), (32, 1024), (64, 256), (128, 64)];

    for (size, traversals) in parameters {
        // Construct Matrices
        let my_impl = SquareMatrix::zero(size);
        let nd_impl = Array2::<Complex<f64>>::zeros((size, size));

        group.bench_with_input(
            BenchmarkId::new("Matrix", size),
            &size,
            |b, &size| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for _ in 0..traversals {
                        for row in 0..size {
                            for col in 0..size {
                                sum += my_impl.get(row, col);
                            }
                        }
                    }

                    black_box(sum);
                }
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Ndarray", size),
            &size,
            |b, &size| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for _ in 0..traversals {
                        for row in 0..size {
                            for col in 0..size {
                                sum += nd_impl[(row, col)];
                            }
                        }
                    }

                    black_box(sum);
                }
            )
        );
    }

    group.finish();
}

// Test traversing Vector sequentially against ndarray::Array1
fn bench_vector_sequential_traversal(c: &mut Criterion) {
    use qsim::linalg::Vector;
    use ndarray::Array1;
    use num_complex::Complex;

    let mut group = c.benchmark_group("Vector Sequential Read");
    group.measurement_time(Duration::from_secs(10));

    // Vector lengths and traversals.
    // Each configuration performs 1,048,576 element reads.
    let parameters = [
        (1, 1_048_576),
        (4, 262_144),
        (16, 65_536),
        (64, 16_384),
        (256, 4_096),
        (1_024, 1_024),
        (4_096, 256),
        (16_384, 64),
        (65_536, 16),
        (262_144, 4),
        (1_048_576, 1),
    ];

    for (size, traversals) in parameters {
        // Construct Matrices
        let my_impl = Vector::zeros(size);
        let nd_impl = Array1::<Complex<f64>>::zeros(size);

        group.bench_with_input(
            BenchmarkId::new("Vector", size),
            &size,
            |b, &size| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for _ in 0..traversals {
                        for i in 0..size {
                            sum += my_impl.get(i);
                        }
                    }

                    black_box(sum);
                }
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Array1", size),
            &size,
            |b, &size| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for _ in 0..traversals {
                        for i in 0..size {
                            sum += nd_impl[i];
                        }
                    }

                    black_box(sum);
                }
            )
        );
    }

    group.finish();
}

// Test traversing Matrix sequentially against ndarray::Array2
fn bench_matrix_random_traversal(c: &mut Criterion) {
    use qsim::linalg::SquareMatrix;

    use ndarray::Array2;
    use num_complex::Complex;
    use rand::{rng, RngExt};

    let mut group = c.benchmark_group("Matrix Random Read");
    group.measurement_time(Duration::from_secs(10));

    // Sizes of N x N matrices.
    let parameters = [(16, 4096), (32, 1024), (64, 256), (128, 64)];

    for (size, accesses) in parameters {
        // Construct Matrices
        let my_impl = SquareMatrix::zero(size);
        let nd_impl = Array2::<Complex<f64>>::zeros((size, size));

        // Generate random coordinates
        let mut rng = rng();
        let coords: Vec<(usize, usize)> = (0..(accesses*size*size))
            .map(|_| (
                rng.random_range(0..size),
                rng.random_range(0..size)
            ))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("Matrix", size),
            &size,
            |b, _| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for &(row, col) in &coords {
                        sum += my_impl.get(row, col);
                    }

                    black_box(sum);
                }
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Ndarray", size),
            &size,
            |b, _| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for &(row, col) in &coords {
                        sum += nd_impl[(row, col)];
                    }

                    black_box(sum);
                }
            )
        );
    }

    group.finish();
}

// Test traversing Vector sequentially against ndarray::Array1
fn bench_vector_random_traversal(c: &mut Criterion) {
    use qsim::linalg::Vector;

    use ndarray::Array1;
    use num_complex::Complex;
    use rand::{rng, RngExt};

    let mut group = c.benchmark_group("Vector Random Read");
    group.measurement_time(Duration::from_secs(10));

    // Vector lengths and traversals.
    // Each configuration performs 1,048,576 element reads.
    let parameters = [
        (1, 1_048_576),
        (4, 262_144),
        (16, 65_536),
        (64, 16_384),
        (256, 4_096),
        (1_024, 1_024),
        (4_096, 256),
        (16_384, 64),
        (65_536, 16),
        (262_144, 4),
        (1_048_576, 1),
    ];

    for (size, accesses) in parameters {
        // Construct Matrices
        let my_impl = Vector::zeros(size);
        let nd_impl = Array1::<Complex<f64>>::zeros(size);

        // Generate random coordinates
        let mut rng = rng();
        let indices: Vec<usize> = (0..(accesses*size))
            .map(|_| rng.random_range(0..size))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("Vector", size),
            &size,
            |b, _| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for &i in &indices {
                        sum += my_impl.get(i);
                    }

                    black_box(sum);
                }
            )
        );

        group.bench_with_input(
            BenchmarkId::new("Array1", size),
            &size,
            |b, _| b.iter(
                || {
                    let mut sum = Complex::<f64>::ZERO;

                    for &i in &indices {
                        sum += nd_impl[i];
                    }

                    black_box(sum);
                }
            )
        );
    }

    group.finish();
}

criterion_group!(
    // Name of function group for benchmarking.
    benches,

    // Targets to add to the function group.
    //bench_kernels_1q_over_target,
    //bench_1q_index_kernel_over_target,
    //bench_matrix_construction,
    bench_vector_construction,
    //bench_matrix_sequential_traversal,
    bench_vector_sequential_traversal,
    //bench_matrix_random_traversal
    bench_vector_random_traversal,
);

// Takes the function group and expands into the program's entry point.
criterion_main!(benches);