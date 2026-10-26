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
    AxisScale,
    BenchmarkId,
    Criterion,
    PlotConfiguration,
    criterion_group,
    criterion_main,
};
use num_complex::Complex;
use qsim::statevector::Config;
use rand::{RngExt, rng};

#[allow(deprecated)]
use qsim::{
    api::Instruction,
    kernels::{
        apply_1q_kronecker,
        apply_1q_strided,
        apply_c2q_kronecker,
        apply_c2q_strided,
        apply_1q_avx,
        apply_1q_fma
    },
    linalg::{SquareMatrix, Vector, linear_map, matrix},
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

// BENCHMARKS EXCLUDED FROM MAIN RESULTS

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

    group.finish();
}
