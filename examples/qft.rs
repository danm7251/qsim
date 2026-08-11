use std::f64::consts::PI;

#[cfg(feature = "trace")]
#[path ="../src/trace.rs"]
mod trace;

use qsim::{api::Instruction, state::State};


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
    state.execute(Instruction::X { q: 1 }).unwrap();

    // Apply QFT circuit.
    let subcircuit: Vec<Instruction> = construct_qft(n);
    for g in subcircuit.clone() {
        state.execute(g).unwrap();
    }

    // Apply QFT circuit again.
    for g in subcircuit {
        state.execute(g).unwrap();
    }

    // State should be back in |010>.
    // `println!("{:?}", state.amplitudes());`
}

fn construct_qft(n: usize) -> Vec<Instruction> {
    let mut circuit = Vec::<Instruction>::new();

    if n == 1 {
        circuit.push(Instruction::H { q: 0 });
        return circuit;
    }

    // For each qubit in the circuit
    for i in 0..n {
        circuit.push(Instruction::H { q: i });
        // For every superior qubit the target qubit has
        for j in (i + 1)..n {
            circuit.push(Instruction::CRP { q_c: j, q_t: i, phi: PI / (1 << (j - i)) as f64 });
        }
    }

    // Construct SWAP gates from CNOTs
    for i in 0..(n / 2) {
        let swap_qubit = n - i - 1;
        circuit.push(Instruction::CNOT { q_c: i, q_t: swap_qubit });
        circuit.push(Instruction::CNOT { q_c: swap_qubit, q_t: i });
        circuit.push(Instruction::CNOT { q_c: i, q_t: swap_qubit });
    }

    circuit
}