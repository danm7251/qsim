// Only compiles if the "bench" feature is enabled since otherwise many Qsim functions are private.
#![cfg(feature = "bench")]

use std::{fs, hint::black_box};

use dhat::Profiler;
use ndarray::{Array1, Array2};
use num_complex::Complex;

use qsim::{
    api::Instruction,
    legacy::{LegacyState, gates::Gate},
    linalg::{SquareMatrix, Vector, linear_map, matrix},
    state::State,
};

mod common;
use common::{construct_qft_for_current, construct_qft_for_legacy};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const OUTPUT_PATH: &'static str = "target/dhat";

/// A collection of related DHAT benchmark cases.
struct BenchGroup {
    name: &'static str,
    active: bool,
    cases: Vec<BenchCase>,
}

/// A single DHAT benchmark case with a name and executable workload.
struct BenchCase {
    name: String,
    bench: Box<dyn FnOnce()>,
}

fn main() {
    for group in benchmarks() {
        if !group.active {
            continue;
        }

        let group_path = format!("{}/{}", OUTPUT_PATH, group.name);

        // Creates the group's output directory.
        fs::create_dir_all(&group_path)
            .unwrap_or_else(|e| panic!("Failed to create {group_path}: {e}"));

        for case in group.cases {
            // Create a separate DHAT output file for each case.
            let filename = format!("{}/{}.json", group_path, case.name);

            // Each profiler represents a single DHAT run.
            let _profiler = Profiler::builder().file_name(filename).build();

            // Run the benchmark case.
            (case.bench)();
        }
    }
}

/// Constructs the DHAT benchmark groups and cases.
fn benchmarks() -> Vec<BenchGroup> {
    // Non-parameterised benchmark groups.
    let mut benches = vec![
        BenchGroup {
            name: "Matrix Vector Multiply Size 2",
            active: false,
            cases: vec![
                {
                    let qsim_vector = Vector::zeros(2);
                    let qsim_matrix = SquareMatrix::zero(2);

                    BenchCase {
                        name: "qsim".into(),
                        bench: Box::new(move || {
                            let res = linear_map(&qsim_matrix, &qsim_vector);
                            black_box(res);
                        }),
                    }
                },
                {
                    let ndarray_vector = Array1::<Complex<f64>>::zeros(2);
                    let ndarray_matrix = Array2::<Complex<f64>>::zeros((2, 2));

                    BenchCase {
                        name: "ndarray".into(),
                        bench: Box::new(move || {
                            let res = ndarray_matrix.dot(&ndarray_vector);
                            black_box(res);
                        }),
                    }
                },
            ],
        }
    ];

    // Parameterised benchmark groups.

    // LEGACY VS CURRENT STATEVECTOR QFT PERFORMANCE

    let parameters: [usize; 8] = [2, 4, 6, 8, 10, 12, 14, 16];

    let mut cases = Vec::<BenchCase>::new();
    for n in parameters {
        cases.push({
            let circuit = construct_qft_for_current(n);

            BenchCase {
                name: format!("current-{n}"),
                bench: Box::new(move || current_qft_execution(n, circuit)),
            }
        });

        cases.push({
            let circuit = construct_qft_for_legacy(n);

            BenchCase {
                name: format!("legacy-{n}"),
                bench: Box::new(move || legacy_qft_execution(n, circuit)),
            }
        });
    }

    benches.push(
        BenchGroup {
            name: "QFT Statevector Performance",
            active: false,
            cases,
        }
    );

    // 1Q INDEX VS STRIDED KERNEL

    let parameters: [usize; 8] = [2, 4, 6, 8, 10, 12, 14, 16];

    let mut cases = Vec::<BenchCase>::new();
    for n in parameters {
        cases.push({
            let mut state = State::zero(n).unwrap();
            let matrix = matrix::x();

            BenchCase {
                name: format!("indexed-{n}"),
                bench: Box::new(move || {
                    // Black box unneeded.
                    state.apply_1q(n / 2, &matrix).unwrap();
                }),
            }
        });

        cases.push({
            let mut state = State::zero(n).unwrap();
            let matrix = matrix::h();

            BenchCase {
                name: format!("strided-{n}"),
                bench: Box::new(move || {
                    // Black box unneeded.
                    // If `linear_map()` is inlined application becomes allocation free.
                    state.apply_1q_strided(0, &matrix).unwrap();
                }),
            }
        });
    }

    benches.push(
        BenchGroup {
            name: "1Q Index & Strided Kernel Performance",
            active: false,
            cases
        }
    );

    // C2Q INDEX VS STRIDED KERNEL

    let parameters: [usize; 7] = [4, 6, 8, 10, 12, 14, 16];

    let mut cases = Vec::<BenchCase>::new();
    for n in parameters {
        // Keep the control/target separation fixed as n scales.
        let control = (n / 2) - 1;
        let target = (n / 2) + 1;

        cases.push({
            let mut state = State::zero(n).unwrap();
            let matrix = matrix::h();

            BenchCase {
                name: format!("indexed-{n}"),
                bench: Box::new(move || {
                    // Black box unneeded.
                    state.apply_c2q(control, target, &matrix).unwrap();
                }),
            }
        });

        cases.push({
            let mut state = State::zero(n).unwrap();
            let matrix = matrix::h();

            BenchCase {
                name: format!("strided-{n}"),
                bench: Box::new(move || {
                    // Black box unneeded.
                    // If `linear_map()` is inlined application becomes allocation free.
                    state.apply_c2q_strided(control, target, &matrix).unwrap();
                }),
            }
        });
    }

    benches.push(
        BenchGroup {
            name: "C2Q Index & Strided Kernel Performance",
            active: true,
            cases
        }
    );

    benches
}

// Helpers

/// Executes a QFT circuit using the current statevector implementation.
fn current_qft_execution(n: usize, circuit: Vec<Instruction>) {
    let mut state = black_box(State::zero(n).unwrap());
    for i in circuit {
        state.execute(i).unwrap();
    }
}

/// Executes a QFT circuit using the legacy statevector implementation.
fn legacy_qft_execution(n: usize, circuit: Vec<Gate>) {
    let mut state = black_box(LegacyState::zero(n).unwrap());
    for g in circuit {
        state.apply_gate(g).unwrap();
    }
}
