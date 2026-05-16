use ndarray::Array1;
use num_complex::Complex64;

#[cfg(debug_assertions)]
use crate::math_utils::print_matrix;
use crate::{gates::Gate64, math_utils::kron};

pub struct State {
    amplitudes: Array1<Complex64>,
    circuit_size: usize
}

impl State {
    // Creates a 0 state of `circuit_size` qubits.
    pub fn zero(circuit_size: usize) -> Self {
        let mut amplitudes = Array1::zeros(1 << circuit_size);
        amplitudes[0] = Complex64::ONE;

        Self {
            amplitudes,
            circuit_size
        }
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        self.amplitudes.as_slice().expect("Array1 should always be contiguous")
    }

    // Applies `gate` to `target` and updates State.
    pub fn apply_gate(&mut self, target: usize, gate: Gate64) -> Result<(), &str> {
        if target >= self.circuit_size {
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

        #[cfg(debug_assertions)]
        print_matrix(&matrix);

        self.amplitudes = matrix.dot(&self.amplitudes);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{f64::consts::SQRT_2};

    use crate::math_utils::C64;

    use super::*;

    #[test]
    fn amplitudes_len_is_circuit_size_to_power_of_two() {
        assert_eq!(State::zero(1).amplitudes().len(), 2);
        assert_eq!(State::zero(3).amplitudes().len(), 8);
        assert_eq!(State::zero(8).amplitudes().len(), 256);
    }

    #[test]
    fn zero_state_is_normalised() {
        let state = State::zero(4);
        let total: f64 = state.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        // Exact equalities will hold for a zero state.
        assert_eq!(total, 1.0);
    }

    #[test]
    fn h_creates_superposition() {
        let mut state = State::zero(1);
        state.apply_gate(0, Gate64::H).expect("Failed to apply Gate64::H");
        let amps = state.amplitudes();
        let expected = C64(1.0 / SQRT_2, 0.);
        // Exact equalities will hold for one gate on a zero state.
        assert_eq!(amps[0], expected);
        assert_eq!(amps[1], expected);
    }
}