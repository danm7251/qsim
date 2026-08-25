// Only compiles if the `bench` feature is enabled since otherwise many qsim functions are private.
#![cfg(feature = "bench")]

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{array, Array1, Array2};
use num_complex::Complex;
use rand::{rng, RngExt};

use qsim::{
    legacy::{LegacyState, gates::Gate},
    linalg::{SquareMatrix, Vector, linear_map},
    state::State,
};

mod common;
use common::{construct_qft_for_current, construct_qft_for_legacy};

// Active benchmarks.
criterion_group!(
    benches,
    bench_legacy_vs_current_state_with_qft
);

criterion_main!(benches);

// KERNEL COMPARISONS

/// Compares the index and Kronecker-product implementations of a
/// single-qubit gate across different target qubits.
#[allow(deprecated, unused)]
fn bench_1q_kernels_index_vs_kronecker(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Index vs Kronecker");
    group.measurement_time(Duration::from_secs(8));

    let n = 8;

    for target in 0..n {
        let gate = Gate::H { target };

        let mut state = LegacyState::zero(n).unwrap();
        group.bench_with_input(BenchmarkId::new("Index", target), &target, |b, target| {
            b.iter(|| state.apply_1q_index(*target, gate.matrix()))
        });

        let mut state = LegacyState::zero(n).unwrap();
        group.bench_with_input(
            BenchmarkId::new("Kronecker", target),
            &target,
            |b, target| b.iter(|| state.apply_1q_kron(*target, gate.matrix())),
        );
    }

    group.finish();
}

/// Benchmarks the 1-qubit index kernel at the first, middle, and last
/// target qubits of a circuit.
#[allow(unused)]
fn bench_1q_index_kernel_over_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Index Kernel by Target");

    let n = 7;
    let targets = [
        0,
        n / 2,
        n - 1
    ];

    for target in targets {
        let gate = Gate::H { target };

        let mut state = LegacyState::zero(n).unwrap();
        group.bench_with_input(BenchmarkId::new("Target", target), &target, |b, target| {
            b.iter(|| state.apply_1q_index(*target, gate.matrix()))
        });
    }

    group.finish();
}

// QSIM LINEAR ALGEBRA VS NDARRAY

/// Benchmarks construction of square matrices across a range of sizes,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_matrix_zero_initialisation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Zero Initialisation");

    // Sizes of N x N matrices.
    let sizes = [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256
    ];

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, &size| {
            b.iter(|| black_box(SquareMatrix::zero(size)))
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, &size| {
            b.iter(|| black_box(Array2::<Complex<f64>>::zeros((size, size))))
        });
    }

    group.finish();
}

/// Benchmarks construction of vectors across a range of sizes,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_vector_zero_initialisation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Zero Initialisation");

    // Vector sizes.
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
        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, &size| {
            b.iter(|| black_box(Vector::zeros(size)))
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, &size| {
            b.iter(|| black_box(Array1::<Complex<f64>>::zeros(size)))
        });
    }

    group.finish();
}

/// Benchmarks sequential matrix traversal across a range of sizes,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_matrix_sequential_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Sequential Read");
    group.measurement_time(Duration::from_secs(10));

    // Matrix dimensions and traversal counts.
    // Each configuration performs 1,048,576 element reads.
    let parameters = [
        (16, 4_096),
        (32, 1_024),
        (64, 256),
        (128, 64)
    ];

    for (size, traversals) in parameters {
        let qsim_matrix = SquareMatrix::zero(size);
        let ndarray_matrix = Array2::<Complex<f64>>::zeros((size, size));

        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, &size| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for _ in 0..traversals {
                    for row in 0..size {
                        for col in 0..size {
                            sum += qsim_matrix.get(row, col);
                        }
                    }
                }

                black_box(sum);
            })
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, &size| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for _ in 0..traversals {
                    for row in 0..size {
                        for col in 0..size {
                            sum += ndarray_matrix[(row, col)];
                        }
                    }
                }

                black_box(sum);
            })
        });
    }

    group.finish();
}

/// Benchmarks sequential vector traversal across a range of sizes,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_vector_sequential_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Sequential Read");
    group.measurement_time(Duration::from_secs(10));

    // Vector lengths and traversal counts.
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
        let qsim_vector = Vector::zeros(size);
        let ndarray_vector = Array1::<Complex<f64>>::zeros(size);

        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, &size| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for _ in 0..traversals {
                    for i in 0..size {
                        sum += qsim_vector.get(i);
                    }
                }

                black_box(sum);
            })
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, &size| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for _ in 0..traversals {
                    for i in 0..size {
                        sum += ndarray_vector[i];
                    }
                }

                black_box(sum);
            })
        });
    }

    group.finish();
}

/// Benchmarks random matrix element access across a range of sizes,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_matrix_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Random Read");
    group.measurement_time(Duration::from_secs(10));

    // Matrix sizes and number of random reads.
    // Each configuration performs 1,048,576 element reads.
    let parameters = [
        (16, 4096),
        (32, 1024),
        (64, 256),
        (128, 64)
    ];

    for (size, num_accesses) in parameters {
        let qsim_matrix = SquareMatrix::zero(size);
        let ndarray_matrix = Array2::<Complex<f64>>::zeros((size, size));

        // Generate random coordinates for each access.
        let mut rng = rng();
        let coords: Vec<(usize, usize)> = (0..(num_accesses * size * size))
            .map(|_| (rng.random_range(0..size), rng.random_range(0..size)))
            .collect();

        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for &(row, col) in &coords {
                    sum += qsim_matrix.get(row, col);
                }

                black_box(sum);
            })
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for &(row, col) in &coords {
                    sum += ndarray_matrix[(row, col)];
                }

                black_box(sum);
            })
        });
    }

    group.finish();
}

/// Benchmarks random vector element access across a range of lengths,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_vector_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Random Read");
    group.measurement_time(Duration::from_secs(10));

    // Vector lengths and number of random reads.
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

    for (size, num_accesses) in parameters {
        let qsim_vector = Vector::zeros(size);
        let ndarray_vector = Array1::<Complex<f64>>::zeros(size);

        // Generate random coordinates for each access.
        let mut rng = rng();
        let indices: Vec<usize> = (0..(num_accesses * size))
            .map(|_| rng.random_range(0..size))
            .collect();

        group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for &i in &indices {
                    sum += qsim_vector.get(i);
                }

                black_box(sum);
            })
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, _| {
            b.iter(|| {
                let mut sum = Complex::<f64>::ZERO;

                for &i in &indices {
                    sum += ndarray_vector[i];
                }

                black_box(sum);
            })
        });
    }

    group.finish();
}

/// Benchmarks 2x2 matrix-vector multiplication,
/// comparing the qsim and ndarray implementations.
#[allow(unused)]
fn bench_matrix_vector_mul_on_pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix-Vector Multiplication");

    let size = 2;

    // Construct Vectors and Matrices.
    let mut qsim_vector = Vector::zeros(size);
    let mut qsim_matrix = SquareMatrix::zero(size);
    let mut ndarray_vector = Array1::<Complex<f64>>::zeros(size);
    let mut ndarray_matrix = Array2::<Complex<f64>>::zeros((size, size));

    // Generate random values.
    let mut rng = rng();
    for i in 0..size {
        let random_complex = Complex::<f64>::new(rng.random(), rng.random());

        *qsim_vector.get_mut(i) = random_complex;
        ndarray_vector[i] = random_complex;

        for j in 0..size {
            let random_complex = Complex::<f64>::new(rng.random(), rng.random());

            *qsim_matrix.get_mut(i, j) = random_complex;
            ndarray_matrix[(i, j)] = random_complex;
        }
    }

    group.bench_with_input(BenchmarkId::new("qsim", size), &size, |b, _| {
        b.iter(|| {
            let res = linear_map(&qsim_matrix, &qsim_vector);
            black_box(res);
        })
    });

    group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |b, _| {
        b.iter(|| {
            let res = ndarray_matrix.dot(&ndarray_vector);
            black_box(res);
        })
    });

    group.finish();
}

/// Benchmarks different 2x2 matrix constructors, comparing qsim options and ndarray.
#[allow(deprecated, unused)]
fn bench_matrix_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Construction");
    group.measurement_time(Duration::from_secs(10));

    let size = 2;
    let values = [
        [Complex::<f64>::new(0.1, 0.25), Complex::<f64>::new(0.15, 0.5)],
        [Complex::<f64>::new(-0.5, 0.26), Complex::<f64>::new(0.101, 1.5)],
    ];

    group.bench_with_input(BenchmarkId::new("qsim/zero + fill", size), &size, |b, _| {
        b.iter(|| {
            let mut res = SquareMatrix::zero(size);
            for (i, row) in values.iter().enumerate() {
                for (j, elem) in row.iter().enumerate() {
                    *res.get_mut(i, j) = *elem;
                }
            }
            black_box(res);
        })
    });

    group.bench_with_input(BenchmarkId::new("qsim/from_array", size), &size, |b, _| {
        b.iter(|| {
            let res = SquareMatrix::from_array([
                [(0.1, 0.25), (0.15, 0.5)],
                [(-0.5, 0.26), (0.101, 1.5)],
            ]);
            black_box(res);
        })
    });

    group.bench_with_input(
        BenchmarkId::new("qsim/from_array (legacy)", size),
        &size,
        |b, _| {
            b.iter(|| {
                let res = SquareMatrix::from_array_2(values);
                black_box(res);
            })
        },
    );

    group.bench_with_input(BenchmarkId::new("ndarray/array!", size), &size, |b, _| {
        b.iter(|| {
            let res = array![
                [Complex::<f64>::new(0.1, 0.25), Complex::<f64>::new(0.15, 0.5)],
                [Complex::<f64>::new(-0.5, 0.26), Complex::<f64>::new(0.101, 1.5)]
            ];
            black_box(res);
        })
    });

    group.finish();
}

/// Benchmarks the existing `state` implementation against the new
/// `new_state` implementation by executing equivalent QFT circuits
/// at different qubit counts.
#[allow(unused)]
fn bench_legacy_vs_current_state_with_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("QFT Statevector Performance");
    group.measurement_time(Duration::from_secs(30));

    // Size of benchmark
    let parameters: [usize; 10] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

    for n in parameters {
        let legacy_circuit = construct_qft_for_legacy(n);
        let current_circuit = construct_qft_for_current(n);

        group.bench_with_input(BenchmarkId::new("legacy", n), &n, |b, _| {
            b.iter(|| {
                let mut state = black_box(LegacyState::zero(n).unwrap());
                for g in &legacy_circuit {
                    state.apply_gate(*g).unwrap();
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("current", n), &n, |b, _| {
            b.iter(|| {
                let mut state = black_box(State::zero(n).unwrap());
                for i in &current_circuit {
                    state.execute(*i).unwrap();
                }
            })
        });
    }

    group.finish();
}
