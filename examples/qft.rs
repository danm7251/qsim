use std::f64::consts::PI;

#[cfg(feature = "trace")]
#[path ="../src/trace.rs"]
mod trace;

use qsim::{gates::Gate, state::State};


/// AN implementation of the 3-qubit Quantum Fourier Transform (QFT).
/// The QFT is the quantum analogue of the Discrete Fourier Transform (DFT), the DFT takes a sequence of numbers,
/// and decomposes them into frequency components. QFT does the same thing but exponentially faster,
/// allowing us to transform the basis states.

fn main() {
    #[cfg(feature = "trace")]
    let _guard = trace::init_tracing();

    let n: usize = 10;

    // Create a |010> state.
    let mut state = State::zero(n).unwrap();
    state.apply_gate(Gate::X { target: 1 }).unwrap();

    // Apply QFT circuit.
    let subcircuit: Vec<Gate> = construct_qft(n);
    for g in subcircuit.clone() {
        state.apply_gate(g).unwrap();
    }

    // Apply QFT circuit again.
    for g in subcircuit {
        state.apply_gate(g).unwrap();
    }

    // State should be back in |010>.
    // `println!("{:?}", state.amplitudes());`
}

fn construct_qft(n: usize) -> Vec<Gate> {
    let mut circuit = Vec::<Gate>::new();

    if n == 1 {
        circuit.push(Gate::H { target: 0 });
        return circuit;
    }

    // For each qubit in the circuit
    for i in 0..n {
        circuit.push(Gate::H { target: i });
        // For every superior qubit the target qubit has
        for j in (i + 1)..n {
            circuit.push(Gate::CRP { control: j, target: i, phi: PI / (1 << (j - i)) as f64 });
        }
    }

    // Construct SWAP gates from CNOTs
    for i in 0..(n / 2) {
        let swap_qubit = n - i - 1;
        circuit.push(Gate::CNOT { control: i, target: swap_qubit });
        circuit.push(Gate::CNOT { control: swap_qubit, target: i });
        circuit.push(Gate::CNOT { control: i, target: swap_qubit });
    }

    circuit
}