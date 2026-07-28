// Only compiles if the "bench" feature is enabled since otherwise many Qsim functions are private.
#![cfg(feature = "bench")]

use std::fs;

use dhat::Profiler;
use qsim::{gates::Gate, state::State};

// Replaces the default global allocator with the `dhat` allocator.
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// The directory path to the results.
const OUTPUT_PATH: &'static str = "target/dhat";

// Benchmark run entry point.
fn main() {
    // Creates the output directory path if it does not already exist.
    fs::create_dir_all(OUTPUT_PATH)
        .unwrap_or_else(|e| panic!("Failed to create {OUTPUT_PATH}: {e}"));

    // Taking inspiration from table driven testing.
    // If `dhat` allows we can then collect data on many benchmarks individually.
    struct BenchCase {
        title: &'static str,
        // Boxing closures makes the cases nicer to write out.
        bench: Box<dyn Fn()>
    }

    let benchmarks: Vec<BenchCase> = vec![
        BenchCase {
            title: "1_Qubit_Idx_Kernel_n16_t3",
            bench: Box::new(|| bench_1q_idx(16, 3))
        }
    ];

    for case in benchmarks {
        // Output file path and name.
        let filename = format!("{}/{}", OUTPUT_PATH, case.title);

        // Each profiler represents a single `dhat` run.
        let _profiler = Profiler::builder()
            .file_name(filename)
            .build();

        // Unusual syntax to call the closures for each benchmark case.
        (case.bench)();
    }
}


// Benchmarks

// Sets up a state of n qubits and applies a T gate to qubit t.
fn bench_1q_idx(n: usize, t: usize) {
    let mut state = State::zero(n).expect("Failed to initialise state");
    let op = Gate::T { target: t };
    state.apply_1q_index(t, op.matrix()).expect("Failed to apply gate");
}