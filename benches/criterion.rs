//! Criterion benchmarks for qsim.
//!
//! Run with `cargo bench --features bench`.
//!
//! Results are saved under `target/criterion/`. Open
//! `target/criterion/report/index.html` for the full report.

// Only compiles if the `bench` feature is enabled since otherwise many qsim functions are private.
#![cfg(feature = "bench")]

use std::{hint::black_box, time::Duration};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use ndarray::{Array1, Array2, array};
use num_complex::Complex;
use qsim::statevector::Config;
use rand::{RngExt, rng};

#[allow(deprecated)]
use qsim::{
    api::Instruction,
    kernels::{
        AvxVariant, apply_1q_avx_with_variant, apply_1q_kronecker, apply_1q_strided,
        apply_c2q_kronecker, apply_c2q_strided, apply_1q_avx, apply_1q_fma
    },
    linalg::{SquareMatrix, Vector, linear_map, matrix},
    stabilizer::Stabilizer,
    statevector::Statevector,
};

mod common;
use common::{target_to_stride, zero_amplitudes};

// Active benchmarks.
criterion_group!(
    benches,
    avx_and_fma_against_avx_over_qubits
);

criterion_main!(benches);

// MAIN RESULTS

// HADAMARD: FULL-SYSTEM MATRIX KERNEL VS PORTABLE DPM KERNEL

fn avx_and_fma_against_avx_over_qubits(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("AVX+FMA vs AVX Performance");
    let config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(config);

    let circuit_sizes = (3..20).step_by(4);
    let circuit = vec![
        Instruction::H { q: 1 },
        Instruction::X { q: 2 },
    ];
    
    for n in circuit_sizes {
        let config = Config { avx: true, fma: false };
        let mut state = Statevector::zero_with_config(n, config).unwrap();
        group.bench_with_input(BenchmarkId::new("AVX only", n), &n, |b, _| {
            b.iter(|| {
                state.execute_all(&circuit).unwrap();
            });
        });

        let config = Config { avx: true, fma: true };
        let mut state = Statevector::zero_with_config(n, config).unwrap();
        group.bench_with_input(BenchmarkId::new("AVX and FMA", n), &n, |b, _| {
            b.iter(|| {
                state.execute_all(&circuit).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_kron_vs_index_on_hadamard_over_qubits(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("Hadamard Gate Performance: Kronecker Expansion vs Direct Indexing");
    let config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(config);

    let circuit_sizes = (3..14).step_by(2);

    for n in circuit_sizes {
        let stride = target_to_stride(n, n / 2);
        let matrix = matrix::h();

        let mut amplitudes = zero_amplitudes(n);
        #[allow(deprecated)]
        group.bench_with_input(BenchmarkId::new("Kronecker expansion", n), &n, |b, _| {
            b.iter(|| {
                apply_1q_kronecker(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&matrix),
                );
            });
        });

        let mut amplitudes = zero_amplitudes(n);
        group.bench_with_input(BenchmarkId::new("Direct indexing", n), &n, |b, _| {
            b.iter(|| {
                apply_1q_strided(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&matrix),
                );
            });
        });
    }

    group.finish();
}

// CNOT: FULL-SYSTEM MATRIX KERNEL VS PORTABLE DPM KERNEL

fn bench_kron_vs_index_on_cnot_over_qubits(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("CNOT Gate Performance: Kronecker Expansion vs Direct Indexing");
    let config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(config);

    let circuit_sizes = (3..14).step_by(2);

    for n in circuit_sizes {
        let c_stride = target_to_stride(n, 0);
        let t_stride = target_to_stride(n, n / 2);
        let matrix = matrix::x();

        let mut amplitudes = zero_amplitudes(n);
        #[allow(deprecated)]
        group.bench_with_input(BenchmarkId::new("Kronecker expansion", n), &n, |b, _| {
            b.iter(|| {
                apply_c2q_kronecker(
                    black_box(&mut amplitudes),
                    black_box(c_stride),
                    black_box(t_stride),
                    black_box(&matrix),
                );
            });
        });

        let mut amplitudes = zero_amplitudes(n);
        group.bench_with_input(BenchmarkId::new("Direct indexing", n), &n, |b, _| {
            b.iter(|| {
                apply_c2q_strided(
                    black_box(&mut amplitudes),
                    black_box(c_stride),
                    black_box(t_stride),
                    black_box(&matrix),
                );
            });
        });
    }

    group.finish();
}

// HADAMARD OVER NUMBER OF QUBITS: PORTABLE DPM KERNEL VS FMA DPM KERNEL VS AVX DPM KERNEL

fn bench_portable_vs_fma_vs_avx_on_hadamard_over_qubits(c: &mut Criterion) {
    if !is_x86_feature_detected!("fma") || !is_x86_feature_detected!("avx") {
        panic!("FMA and AVX unsupported on host machine!");
    }

    let mut group = c.benchmark_group("Hadamard Gate Performance over N: Portable vs FMA vs AVX");
    let config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(config);

    let circuit_sizes = (3..21).step_by(2);

    for n in circuit_sizes {
        let stride = target_to_stride(n, n / 2);
        let matrix = matrix::h();

        let mut amplitudes = zero_amplitudes(n);
        group.bench_with_input(BenchmarkId::new("Portable", n), &n, |b, _| {
            b.iter(|| {
                apply_1q_strided(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&matrix),
                );
            });
        });

        let mut amplitudes = zero_amplitudes(n);
        unsafe {
            group.bench_with_input(BenchmarkId::new("FMA", n), &n, |b, _| {
                b.iter(|| {
                    apply_1q_fma(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&matrix),
                    );
                });
            });
        }

        let mut amplitudes = zero_amplitudes(n);
        unsafe {
            group.bench_with_input(BenchmarkId::new("AVX", n), &n, |b, _| {
                b.iter(|| {
                    apply_1q_avx(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&matrix),
                    );
                });
            });
        }
    }

    group.finish();
}

// HADAMARD OVER TARGET QUBIT: PORTABLE DPM KERNEL VS FMA DPM KERNEL VS AVX DPM KERNEL

fn bench_portable_vs_fma_vs_avx_on_hadamard_over_targets(c: &mut Criterion) {
    if !is_x86_feature_detected!("fma") || !is_x86_feature_detected!("avx") {
        panic!("FMA and AVX unsupported on host machine!");
    }

    let mut group =
        c.benchmark_group("Hadamard Gate Performance over T at 17: Portable vs FMA vs AVX");
    let n = 17;

    for t in 0..n {
        let stride = target_to_stride(n, t);
        let matrix = matrix::h();

        let mut amplitudes = zero_amplitudes(n);
        group.bench_with_input(BenchmarkId::new("Portable", t), &t, |b, _| {
            b.iter(|| {
                apply_1q_strided(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&matrix),
                );
            });
        });

        let mut amplitudes = zero_amplitudes(n);
        unsafe {
            group.bench_with_input(BenchmarkId::new("FMA", t), &t, |b, _| {
                b.iter(|| {
                    apply_1q_fma(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&matrix),
                    );
                });
            });
        }

        if stride > 1 {
            let mut amplitudes = zero_amplitudes(n);
            unsafe {
                group.bench_with_input(BenchmarkId::new("AVX", t), &t, |b, _| {
                    b.iter(|| {
                        apply_1q_avx(
                            black_box(&mut amplitudes),
                            black_box(stride),
                            black_box(&matrix),
                        );
                    });
                });
            }
        }
    }

    group.finish();
}

// CLIFFORD GATES: STATEVECTOR VS STABILIZER SIMULATORS

fn bench_statevector_vs_stabilizer_over_qubits(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clifford Circuit Performance: Statevector vs Stabilizer");

    let config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(config);

    let n_range = (3..21).step_by(2);
    let circuit = vec![
        Instruction::X { q: 1 },
        Instruction::CNOT { q_c: 1, q_t: 0 }
    ];

    for n in n_range {

        let mut state = black_box(Statevector::zero(n).unwrap());
        group.bench_with_input(BenchmarkId::new("Statevector", n), &n, |b, &_n| {
            b.iter(|| {
                state.execute_all(black_box(&circuit)).unwrap();
            });
        });

        let mut state = black_box(Stabilizer::zero(n).unwrap());
        group.bench_with_input(BenchmarkId::new("Stabilizer", n), &n, |b, &_n| {
            b.iter(|| {
                state.execute_all(black_box(&circuit)).unwrap();
            });
        });
    }

    group.finish();
}

// BENCHMARKS EXCLUDED FROM MAIN RESULTS

/// Compares the generic kernel and AVX variants across target qubits.
#[allow(unused)]
fn bench_generic_vs_avx_over_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Kernel Variants by Target");
    group.measurement_time(Duration::from_secs(10));

    let n = 16;
    let gate = matrix::y();

    // The final target has stride 1, which the SIMD kernel cannot process.
    let targets = [0, n / 2, n - 2];

    for target in targets {
        let stride = 1 << (n - target - 1);
        let mut amplitudes = zero_amplitudes(n);

        group.bench_with_input(BenchmarkId::new("generic", target), &target, |b, _| {
            b.iter(|| {
                apply_1q_strided(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&gate),
                );
            });
        });

        let variants = [
            ("avx-scalar", AvxVariant::Scalar),
            ("avx-portable", AvxVariant::Portable),
        ];

        for (name, variant) in variants {
            let mut amplitudes = zero_amplitudes(n);

            group.bench_with_input(BenchmarkId::new(name, target), &target, |b, _| {
                b.iter(|| unsafe {
                    apply_1q_avx_with_variant(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&gate),
                        variant,
                    );
                });
            });
        }
    }

    group.finish();
}

/// Compares the generic kernel and AVX variants across state-vector sizes.
#[allow(unused)]
fn bench_generic_vs_avx_over_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("1Q Kernel Variants by Qubits");
    group.measurement_time(Duration::from_secs(10));

    let qubit_counts = [4, 8, 12, 16, 20];
    let target = 0;
    let gate = matrix::y();

    for n in qubit_counts {
        let stride = 1 << (n - target - 1);
        let mut amplitudes = zero_amplitudes(n);

        group.bench_with_input(BenchmarkId::new("generic", n), &n, |b, _| {
            b.iter(|| {
                apply_1q_strided(
                    black_box(&mut amplitudes),
                    black_box(stride),
                    black_box(&gate),
                );
            });
        });

        let variants = [
            ("avx-scalar", AvxVariant::Scalar),
            ("avx-portable", AvxVariant::Portable),
        ];

        for (name, variant) in variants {
            let mut amplitudes = zero_amplitudes(n);

            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| unsafe {
                    apply_1q_avx_with_variant(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&gate),
                        variant,
                    );
                });
            });
        }
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
    let sizes = [2, 4, 8, 16, 32, 64, 128, 256];

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
        1, 4, 16, 64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576,
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
    let parameters = [(16, 4_096), (32, 1_024), (64, 256), (128, 64)];

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
    let parameters = [(16, 4096), (32, 1024), (64, 256), (128, 64)];

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
        [
            Complex::<f64>::new(0.1, 0.25),
            Complex::<f64>::new(0.15, 0.5),
        ],
        [
            Complex::<f64>::new(-0.5, 0.26),
            Complex::<f64>::new(0.101, 1.5),
        ],
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
                [
                    Complex::<f64>::new(0.1, 0.25),
                    Complex::<f64>::new(0.15, 0.5)
                ],
                [
                    Complex::<f64>::new(-0.5, 0.26),
                    Complex::<f64>::new(0.101, 1.5)
                ]
            ];
            black_box(res);
        })
    });

    group.finish();
}
