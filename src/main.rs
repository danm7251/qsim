#[cfg(feature = "trace")]
mod trace;

use qsim::{api::Instruction, statevector::StateVector};

const DEBUG: bool = false;

fn main() {
    #[cfg(feature = "trace")]
    let _guard = trace::init_tracing();

    let num_q = 12;
    let mut state = StateVector::zero(num_q).unwrap();
    state.execute(Instruction::X { q: 2 }).unwrap();
}

#[allow(unused)]
fn show_norm(state: &StateVector) {
    if DEBUG { println!("L2 Norm: {:.2}", state.norm()) }
}

#[allow(unused)]
fn show_measure(state: &mut StateVector, target: usize) {
    if DEBUG {
        let result = if state.measure(target).unwrap() { 1 } else { 0 };
        println!("Q{} = {}", target, result)
    }
}

#[allow(unused)]
fn show_state(state: &StateVector) {
    if DEBUG {
        for (i, amp) in state.amplitudes().iter().enumerate() {
            println!("Amplitude {} = {}", i, amp)
        }
    }
}