#![cfg(feature = "bench")]

use dhat::{Alloc, Profiler};
use qsim::{gates::Gate, state::State};

#[global_allocator]
static ALLOC: Alloc = Alloc;

fn main() {
    struct BenchCase {
        title: &'static str,
        bench: Box<dyn Fn()>
    }

    let benchmarks: Vec::<BenchCase> = vec![
        BenchCase {
            title: "1 Qubit Idx Kernel (n=16, t=3)",
            bench: Box::new(|| bench_1q_idx(16, 3))
        }
    ];

    for case in benchmarks {
        let _profiler = Profiler::new_heap();
        (case.bench)();
    }
}

fn bench_1q_idx(n: usize, t: usize) {
    let mut state = State::zero(n).expect("Failed to initialise state");
    let op = Gate::T { target: t };
    state.apply_1q_index(t, op.matrix()).expect("Failed to apply gate");
}