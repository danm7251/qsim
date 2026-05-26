use ndarray::{Array1, Array2};
use num_complex::{Complex64, ComplexFloat};
use rand::random;

use crate::{gates::Gate, math_utils::kron};

pub struct State {
    amplitudes: Array1<Complex64>, // Big endian, for now.
    n: usize
}

impl State {
    /// Creates a 0 state of `circuit_size` qubits.
    pub fn zero(circuit_size: usize) -> Self {
        let mut amplitudes = Array1::zeros(1 << circuit_size);
        amplitudes[0] = Complex64::ONE;

        Self {
            amplitudes,
            n: circuit_size
        }
    }

    // Getters
    /// Returns a slice of all the probability amplitudes tracked by the state.
    /// Can panic if the internal array is not contiguous or in standard order.
    pub fn amplitudes(&self) -> &[Complex64] {
        self.amplitudes.as_slice().expect("Array1 should always be contiguous")
    }

    /// Returns the number of qubits that make up the entire state.
    pub fn circuit_size(&self) -> usize {
        self.n
    }

    /// Returns the L2 norm of the entire state.
    pub fn norm(&self) -> f64 {
        self.amplitudes.iter().map(|amp| amp.norm_sqr()).sum()
    }

    /// Pretty print the probability distribution of the target qubit.
    pub fn print_probability(self, target: usize) {
        let prob_0: f64 = self.amplitudes
            .iter() // Gives us a iterator of references
            .enumerate() // Gives us the index of each reference
            .filter(|(i, _)| index_is_zero(self.n, target, *i)) // We select the indexes we want
            .map(|(_, a)| a.norm_sqr().re()) // We transform them to square norms
            .sum(); // We recieve the total
        let prob_1 = 1. - prob_0;
        
        println!("q{} = {:.2}% |0|, {:.2}% |1|", target, prob_0 * 100., prob_1 * 100.)
    }

    pub fn apply_gate(&mut self, gate: Gate) -> Result<(), &str> {
        match gate {
            Gate::CNOT { control, target } => self.apply_cnot(control, target),
            Gate::I => Err("Why would you use this"),
            Gate::X { target } => self.apply_1q_gate(target, gate.matrix()),
            Gate::Y { target } => self.apply_1q_gate(target, gate.matrix()),
            Gate::Z { target } => self.apply_1q_gate(target, gate.matrix()),
            Gate::H { target } => self.apply_1q_gate(target, gate.matrix()),
            Gate::S { target } => self.apply_1q_gate(target, gate.matrix()),
            Gate::T { target } => self.apply_1q_gate(target, gate.matrix())
        }
    }

    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), &str> {
        if control == target {
            return Err("Control and target must be distinct qubits")
        } else if control >= self.n || target >= self.n {
            return Err("Control and target must be an existing qubit")
        }

        // A CNOT gate works by swapping pairs of amplitudes in which the control bit is one but the target bit differs.
        for index_low in 0..self.amplitudes.len() {
            // We find every amplitude where c == 1 && t == 0.
            if !index_is_zero(self.n, control, index_low) && index_is_zero(self.n, target, index_low) {
                // We calculate the corresponding amplitude index with i2 = i1 + 2^(n - k - 1).
                let index_high = index_low + (1 << self.n - target - 1);
                // We swap the contents
                self.amplitudes.swap(index_low, index_high);
            }
        }

        Ok(())
    }

    // Functionality
    // Naive implementation of `apply_gate()`.
    /// Applies `gate` to `target`, updating the state.
    fn apply_1q_gate(&mut self, target: usize, gate_matrix: Array2<Complex64>) -> Result<(), &str> {
        if target >= self.n {
            return Err("Target qubit does not exist")
        }

        let identity = Gate::I.matrix();

        // Construct full circuit matrix and apply
        let mut matrix = if target == 0 { gate_matrix.clone() } else { identity.clone() };

        for q in 1..self.n {
            if q == target {
                matrix = kron(&matrix, &gate_matrix)
            } else {
                matrix = kron(&matrix, &identity)
            }
        }

        self.amplitudes = matrix.dot(&self.amplitudes);

        Ok(())
    }

    pub fn measure(&mut self, target: usize) -> bool {
        let circuit_size = self.n;

        // Sum the magnitudes of all amplitudes where q[target] = 0
        let mut prob_0 = 0.;
        for (index, amp) in self.amplitudes.iter().enumerate() {
            if index_is_zero(circuit_size, target, index) {
                prob_0 += amp.norm_sqr().re();
            }
        }

        // Determine outcome based on where a random number lands
        let outcome = random::<f64>() > prob_0;
        println!("Probability of 0 = {}", prob_0);

        // Collapse state
        for (index, amp) in self.amplitudes.iter_mut().enumerate() {
            // If the index matches the outcome then renormalise otherwise set to zero
            if index_is_zero(circuit_size, target, index) {
                if outcome {
                    println!("Setting amp {} to ZERO", index);
                    *amp = Complex64::ZERO;
                } else {
                    println!("Setting amp {} to sqrt(prob_0)", index);
                    *amp /= prob_0.sqrt();
                }
            } else {
                if outcome {
                    println!("Setting amp {} to sqrt(prob_1)", index);
                    *amp /= (1. - prob_0).sqrt(); // TODO: Should be prob_1 I think check later.
                } else {
                    println!("Setting amp {} to ZERO", index);
                    *amp = Complex64::ZERO;
                }
            }
        }

        outcome
    }
}

fn index_is_zero(circuit_size: usize, target: usize, index: usize) -> bool {
    let stride = 1 << (circuit_size - target - 1);
    let block_size = 2 * stride;

    // This gives the position relative to the pattern.
    let pos = index % block_size;

    // The range includes 0 and excludes stride
    if (0..stride).contains(&pos) {
        return true
    }

    false
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
        state.apply_gate(Gate::H {target: 0}).expect("Failed to apply Gate::H");
        let amps = state.amplitudes();
        let expected = C64(1.0 / SQRT_2, 0.);
        // Exact equalities will hold for one gate on a zero state.
        assert_eq!(amps[0], expected);
        assert_eq!(amps[1], expected);
    }
}