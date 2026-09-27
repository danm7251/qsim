#[cfg(feature = "trace")]
mod trace;

use qsim::{api::Instruction, kernels::apply_1q_strided, state::State};

const DEBUG: bool = false;

fn main() {
    use num_complex::Complex;
    use qsim::linalg::matrix;

    #[cfg(feature = "trace")]
    let _guard = trace::init_tracing();

    let n = 17;
    let t: usize = std::env::args()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();

    let stride =     1 << (n - t - 1);
    let matrix = matrix::h();
    let mut amplitudes = std::hint::black_box(vec![Complex::new(0.0, 0.0); 1 << n]);
    amplitudes[0] = Complex::new(1.0, 0.0);

    for _ in 0..1000 {
        apply_1q_strided(&mut amplitudes, stride, &matrix);
    }
}

#[allow(unused)]
fn show_norm(state: &State) {
    if DEBUG { println!("L2 Norm: {:.2}", state.norm()) }
}

#[allow(unused)]
fn show_measure(state: &mut State, target: usize) {
    if DEBUG {
        let result = if state.measure(target).unwrap() { 1 } else { 0 };
        println!("Q{} = {}", target, result)
    }
}

#[allow(unused)]
fn show_state(state: &State) {
    if DEBUG {
        for (i, amp) in state.amplitudes().iter().enumerate() {
            println!("Amplitude {} = {}", i, amp)
        }
    }
}