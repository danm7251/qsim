use qsim::{gates::Gate, state::State};

const DEBUG: bool = false;

fn main() {
    // Example usage
    let num_qubits = 13;
    let mut state = State::zero(num_qubits).unwrap();
    show_state(&state);
    show_norm(&state);
    state.apply_gate(Gate::H {target: 0}).unwrap();
    show_state(&state);
    show_norm(&state);
    state.apply_gate(Gate::CNOT { control: 0, target: 5 }).unwrap();
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
    if DEBUG { println!("L2 Norm: {:.2}", state.norm()) }
}

fn show_measure(state: &mut State, target: usize) {
    if DEBUG {
        let result = if state.measure(target) { 1 } else { 0 };
        println!("Q{} = {}", target, result)
    }
}

fn show_state(state: &State) {
    if DEBUG {
        for (i, amp) in state.amplitudes().iter().enumerate() {
            println!("Amplitude {} = {}", i, amp)
        }
    }
}