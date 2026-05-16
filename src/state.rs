use ndarray::Array1;
use num_complex::Complex64;

use crate::{gates::Gate64, math_utils::kron};

pub struct State {
    amplitudes: Array1<Complex64>,
    circuit_size: usize
}

impl State {
    pub fn zero(circuit_size: usize) -> Self {
        let mut amplitudes = Array1::zeros(1 << circuit_size);
        amplitudes[0] = Complex64::ONE;

        println!("{}", amplitudes.len());

        Self {
            amplitudes,
            circuit_size
        }
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        return self.amplitudes.as_slice().expect("Array1 should always be contiguous")
    }

    pub fn apply_gate(&mut self, target: usize, gate: Gate64) -> Result<(), &str> {
        if target > self.circuit_size - 1 {
            return Err("Target qubit does not exist")
        }

        let identity = Gate64::I.matrix();

        // Construct full circuit matrix and apply
        let mut matrix = if target == 0 { gate.matrix() } else { identity.clone() };

        for q in 1..self.circuit_size {
            if q == target {
                matrix = kron(&matrix, &gate.matrix())
            } else {
                matrix = kron(&matrix, &identity)
            }
        }
        self.amplitudes = matrix.dot(&self.amplitudes);

        Ok(())
    }
}