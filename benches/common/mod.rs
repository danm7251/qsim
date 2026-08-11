use std::f64::consts::PI;

use qsim::{api::Instruction, legacy::gates::Gate};

pub fn construct_qft_for_legacy(n: usize) -> Vec<Gate> {
    let mut circuit = Vec::<Gate>::new();

    // For each qubit in the circuit
    for i in 0..n {
        circuit.push(Gate::H { target: i });
        // For every superior qubit the target qubit has
        for j in (i + 1)..n {
            circuit.push(Gate::CRP {
                control: j,
                target: i,
                phi: PI / (1 << (j - i)) as f64,
            });
        }
    }

    // Construct SWAP gates from CNOTs
    for i in 0..(n / 2) {
        let swap_qubit = n - i - 1;
        circuit.push(Gate::CNOT {
            control: i,
            target: swap_qubit,
        });
        circuit.push(Gate::CNOT {
            control: swap_qubit,
            target: i,
        });
        circuit.push(Gate::CNOT {
            control: i,
            target: swap_qubit,
        });
    }

    circuit
}

pub fn construct_qft_for_current(n: usize) -> Vec<Instruction> {
    let mut circuit = Vec::<Instruction>::new();

    // For each qubit in the circuit
    for i in 0..n {
        circuit.push(Instruction::H { q: i });
        // For every superior qubit the target qubit has
        for j in (i + 1)..n {
            circuit.push(Instruction::CRP {
                q_c: j,
                q_t: i,
                phi: PI / (1 << (j - i)) as f64,
            });
        }
    }

    // Construct SWAP gates from CNOTs
    for i in 0..(n / 2) {
        let swap_qubit = n - i - 1;
        circuit.push(Instruction::CNOT {
            q_c: i,
            q_t: swap_qubit,
        });
        circuit.push(Instruction::CNOT {
            q_c: swap_qubit,
            q_t: i,
        });
        circuit.push(Instruction::CNOT {
            q_c: i,
            q_t: swap_qubit,
        });
    }

    circuit
}
