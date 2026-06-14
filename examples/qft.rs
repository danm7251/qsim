use std::f64::consts::PI;

use qsim::{gates::Gate, state::State};


/// AN implementation of the 3-qubit Quantum Fourier Transform (QFT).
/// The QFT is the quantum analogue of the Discrete Fourier Transform (DFT), the DFT takes a sequence of numbers,
/// and decomposes them into frequency components. QFT does the same thing but exponentially faster,
/// allowing us to transform the basis states.

fn main() {
    let n: usize = 3;

    let mut state = State::zero(n).unwrap();
    let subcircuit: Vec<Gate> = construct_qft(n);
    for g in subcircuit {
        state.apply_gate(g).unwrap();
    }

    print!("{:?}", state.amplitudes());

}

fn construct_qft(n: usize) -> Vec<Gate> {
    let mut circuit = Vec::<Gate>::new();

    if n == 1 {
        circuit.push(Gate::H { target: 0 });
        return circuit;
    }

    for i in 0..n {
        circuit.push(Gate::H { target: i });
        for j in 0..(n - i - 1) {
            circuit.push(Gate::CRP { control: j + i + 1, target: i, phi: PI / (2*(1 << (j - i))) as f64 });
        }
    }

    println!("{:?}", circuit);

    circuit
}