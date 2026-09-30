// Only compiles if the "bench" feature is enabled since otherwise many Qsim functions are private.
#![cfg(feature = "bench")]

use std::{fs, hint::black_box};

use dhat::Profiler;
use ndarray::{Array1, Array2};
use num_complex::Complex;

use qsim::stabilizer::Stabilizer;
#[allow(deprecated)]
use qsim::{
    api::Instruction,
    kernels::{apply_1q_kronecker, apply_1q_strided, apply_c2q_kronecker, apply_c2q_strided},
    linalg::{SquareMatrix, Vector, linear_map, matrix},
    statevector::Statevector,
};

mod common;
use common::{
    construct_qft, target_to_stride, zero_amplitudes,
};

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
    let mut benches = vec![BenchGroup {
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
    }];

    // Parameterised benchmark groups.

    let circuit_sizes = (3..14).step_by(2);

    let mut cases = Vec::<BenchCase>::new();
    for n in circuit_sizes {
        let stride = target_to_stride(n, n / 2);

        cases.push({
            let mut amplitudes = zero_amplitudes(n);
            let matrix = matrix::h();

            #[allow(deprecated)]
            BenchCase {
                name: format!("Kronecker-expansion-hadamard-{n}"),
                bench: Box::new(move || {
                    apply_1q_kronecker(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&matrix),
                    );
                }),
            }
        });

        cases.push({
            let mut amplitudes = zero_amplitudes(n);
            let matrix = matrix::h();

            BenchCase {
                name: format!("Direct-indexing-hadamard-{n}"),
                bench: Box::new(move || {
                    apply_1q_strided(
                        black_box(&mut amplitudes),
                        black_box(stride),
                        black_box(&matrix),
                    );
                }),
            }
        });
    }

    benches.push(BenchGroup {
        name: "Hadamard Gate Performance: Kronecker Expansion and Direct Indexing",
        active: true,
        cases,
    });

    
    let circuit_sizes = (3..14).step_by(2);

    let mut cases = Vec::<BenchCase>::new();
    for n in circuit_sizes {
        let c_stride = target_to_stride(n, 0);
        let t_stride = target_to_stride(n, n / 2);

        cases.push({
            let mut amplitudes = zero_amplitudes(n);
            let matrix = matrix::x();

            #[allow(deprecated)]
            BenchCase {
                name: format!("Kronecker-expansion-cnot-{n}"),
                bench: Box::new(move || {
                    apply_c2q_kronecker(
                        black_box(&mut amplitudes),
                        black_box(c_stride),
                        black_box(t_stride),
                        black_box(&matrix),
                    );
                }),
            }
        });

        cases.push({
            let mut amplitudes = zero_amplitudes(n);
            let matrix = matrix::x();

            BenchCase {
                name: format!("Direct-indexing-cnot-{n}"),
                bench: Box::new(move || {
                    apply_c2q_strided(
                        black_box(&mut amplitudes),
                        black_box(c_stride),
                        black_box(t_stride),
                        black_box(&matrix),
                    );
                }),
            }
        });
    }

    benches.push(BenchGroup {
        name: "CNOT(CX) Gate Performance: Kronecker Expansion and Direct Indexing",
        active: false,
        cases,
    });

    // Statevector vs Stabilizer memory performance.

    let n_range = (3..21).step_by(2);
    let circuit = vec![
        Instruction::X { q: 1 },
        Instruction::CNOT { q_c: 1, q_t: 0 }
    ];

    let mut cases = Vec::<BenchCase>::new();
    for n in n_range {
        cases.push({
            let mut state = black_box(Statevector::zero(n).unwrap());
            let circuit = circuit.clone();

            BenchCase {
                name: format!("Statevector-clifford-{n}"),
                bench: Box::new(move || {
                    state.execute_all(black_box(&circuit)).unwrap();
                }),
            }
        });

        cases.push({
            let mut state = black_box(Stabilizer::zero(n).unwrap());
            let circuit = circuit.clone();

            BenchCase {
                name: format!("Stabiliser-clifford-{n}"),
                bench: Box::new(move || {
                    state.execute_all(black_box(&circuit)).unwrap();
                }),
            }           
        });
    }

    benches.push(BenchGroup {
        name: "Clifford circuit Statevector vs Stabilizer",
        active: true,
        cases,
    });

    benches
}