use qsim::{math_utils::print_matrix, state::State};

/// Running notes:
/// - When I was using prob_0 in both renorms the norm went 1->0->NaN->NaN each measurement.
/// - When I used prob_1 in the second renorm the norm went 1->NaN->NaN each measurement.

fn main() {
    // Example usage
    let num_qubits = 3;
    let mut state = State::zero(num_qubits);
    show_state(&state);
    show_norm(&state);
    state.apply_gate(1, qsim::gates::Gate64::H).unwrap();
    show_state(&state);
    show_norm(&state);
    show_measure(&mut state, 0);
    show_state(&state);
    show_norm(&state);
    show_measure(&mut state, 1);
    show_state(&state);
    show_norm(&state);
    show_measure(&mut state, 2);
    show_state(&state);
    show_norm(&state);
}

fn show_norm(state: &State) {
    println!("L2 Norm: {:.2}", state.norm())
}

fn show_measure(state: &mut State, target: usize) {
    let result = if state.measure(target) { 1 } else { 0 };
    println!("Q{} = {}", target, result)
}

fn show_state(state: &State) {
    for (i, amp) in state.amplitudes().iter().enumerate() {
        println!("Amplitude {} = {}", i, amp)
    }
}