use num_complex::{Complex64, ComplexFloat};
use rand::random;

use crate::{api::Instruction::{self, *}, linalg::{SquareMatrix, Vector, linear_map, matrix}};

pub struct State {
    amplitudes: Vector, // Big endian, for now.
    n: usize
}

// Thinking about how to handle safety here is important.
// For example circuit_size == 0 should be an invalid state.
// While compile-time safety seems preferable for users writing rawly in the library,
// it would require the use of const generic assertions which would limit the flexibility,
// and bloat the API. It may be better to limit the libraries assertions to runtime,
// and enforce compile time safety in the DSL instead.
// It will depend how feasible the DSL seems later on.

impl State {
    /// Creates a 0 state of `circuit_size` qubits.
    
    #[cfg_attr(feature = "trace", tracing::instrument(name = "Zero State Construction", err))]
    pub fn zero(circuit_size: usize) -> Result<Self, &'static str> {
        if circuit_size == 0 {
            // I will want to implement a custom error type soon.
            return Err("A state with 0 qubits is invalid");
        }

        let mut amplitudes = Vector::zeros(1 << circuit_size);
        *amplitudes.get_mut(0) = Complex64::ONE;

        Ok(Self {
            amplitudes,
            n: circuit_size
        })
    }

    // Getters
    /// Returns a slice of all the probability amplitudes tracked by the state.
    /// Can panic if the internal array is not contiguous or in standard order.
    pub fn amplitudes(&self) -> &[Complex64] {
        self.amplitudes.as_slice()
    }

    /// Returns the number of qubits that make up the entire state.
    pub fn circuit_size(&self) -> usize {
        self.n
    }

    /// Returns the L2 norm of the entire state.
    pub fn norm(&self) -> f64 {
        self.amplitudes.as_slice().iter().map(|amp| amp.norm_sqr()).sum()
    }

    /// Pretty print the probability distribution of the target qubit.
    pub fn print_probability(&self, target: usize) {
        let prob_0: f64 = self.amplitudes.as_slice()
            .iter() // Gives us a iterator of references
            .enumerate() // Gives us the index of each reference
            .filter(|(i, _)| index_is_zero(self.n, target, *i)) // We select the indexes we want
            .map(|(_, a)| a.norm_sqr().re()) // We transform them to square norms
            .sum(); // We recieve the total
        let prob_1 = 1. - prob_0;
        
        println!("q{} = {:.2}% |0|, {:.2}% |1|", target, prob_0 * 100., prob_1 * 100.)
    }

    // Functionality
    /// Applies a 'Gate' instruction to the state.
    pub fn execute(&mut self, cmd: Instruction) -> Result<(), &'static str> {
        match cmd {
            // Stats
            ViewProb { q } =>  {
                self.print_probability(q);
                return Ok(())
            },
            // Operations
            Measure { q } =>  {
                self.measure(q);
                return Ok(())
            },
            // One Qubit Gates
            X { q } => { 
                return self.apply_1q_index(q, matrix::x())
            },
            Y { q } =>  {
                return self.apply_1q_index(q, matrix::y());
            },
            Z { q } => { 
                return self.apply_1q_index(q, matrix::z());
            },
            H { q } => {
                return self.apply_1q_index(q, matrix::h());
            },
            S { q } => { 
                return self.apply_1q_index(q, matrix::s());
            },
            T { q } => { 
                return self.apply_1q_index(q, matrix::t());
            },
            P { q, phi } => { 
                return self.apply_1q_index(q, matrix::p(phi));
            },
            // Controlled One Qubit Gates
            CNOT { q_c, q_t } => { 
                return self.apply_c2q_index(q_c, q_t, matrix::x());
            },
            CRP { q_c, q_t, phi } => {
                return self.apply_c2q_index(q_c, q_t, matrix::p(phi));
            },
            // Two Qubit Gates
            SWAP { q_1: _, q_2: _ } => unimplemented!("SWAP is unimplemented!"),
            // Subroutines
            QFT => unimplemented!("QFT is unimplemented!"),
        }
    }

    #[cfg_attr(feature = "bench", visibility::make(pub))]
    #[cfg_attr(feature = "trace", tracing::instrument(skip(self, gate_matrix), name = "1 Qubit Gate", err))]
    fn apply_1q_index(&mut self, target: usize, gate_matrix: SquareMatrix) -> Result<(), &'static str> {
        if target >= self.n {
            return Err("Target qubit does not exist")
        }

        for index_low in 0..self.amplitudes.len() {
            if index_is_zero(self.n, target, index_low) {
                // Find index where the bitstring only differs on `target`.
                let index_high = index_low + (1 << self.n - target - 1);
                // Create a pair/subspace.
                let pair = Vector::from_elements([
                    *self.amplitudes.get(index_low),
                    *self.amplitudes.get(index_high)
                ]);
                let updated_pair = linear_map(&gate_matrix, &pair);
                // Replace amplitudes with updated values.
                *self.amplitudes.get_mut(index_low) = *updated_pair.get(0);
                *self.amplitudes.get_mut(index_high) = *updated_pair.get(1);
            }
        }

        Ok(())
    }

    // Controlled two-qubit general
    // While initially I had a specialised CNOT kernel,
    // and often specific implementations of gates are more efficient.
    // Maintaining specialised functions for each gate would be a nightmare.
    #[cfg_attr(feature = "bench", visibility::make(pub))]
    #[cfg_attr(feature = "trace", tracing::instrument(skip(self, gate_matrix), name = "2 Qubit Gate", err))]
    fn apply_c2q_index(&mut self, control: usize, target: usize, gate_matrix: SquareMatrix) -> Result<(), &'static str> {
        if control == target {
            return Err("Control and target must be distinct qubits")
        } else if control >= self.n || target >= self.n {
            return Err("Control and target must be an existing qubit")
        }

        // Iterate over every amplitude (inefficient I know).
        for index_low in 0..self.amplitudes.len() {
            // If control == 1 and target == 0
            if !index_is_zero(self.n, control, index_low) && index_is_zero(self.n, target, index_low) {
                // Find index where the bitstring only differs on `target`.
                let index_high = index_low + (1 << self.n - target - 1);
                // Create a pair/subspace.
                let pair = Vector::from_elements([
                    *self.amplitudes.get(index_low),
                    *self.amplitudes.get(index_high)
                ]);
                // Perform operation.
                let updated_pair = linear_map(&gate_matrix, &pair);
                // Update values
                *self.amplitudes.get_mut(index_low) = *updated_pair.get(0);
                *self.amplitudes.get_mut(index_high) = *updated_pair.get(1);
            }
        }

        Ok(())
    }

    pub fn measure(&mut self, target: usize) -> bool {
        let circuit_size = self.n;

        // Sum the magnitudes of all amplitudes where q[target] = 0
        let mut prob_0 = 0.;
        for (index, amp) in self.amplitudes.as_slice().iter().enumerate() {
            if index_is_zero(circuit_size, target, index) {
                prob_0 += amp.norm_sqr().re();
            }
        }

        // Determine outcome based on where a random number lands
        let outcome = random::<f64>() > prob_0;
        println!("Probability of 0 = {}", prob_0);

        // Collapse state
        for (index, amp) in self.amplitudes.as_mut_slice().iter_mut().enumerate() {
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
    use std::f64::consts::{PI, FRAC_1_SQRT_2};

    use crate::math_utils::C64;

    use super::*;

    #[test]
    fn amplitudes_len_is_circuit_size_to_power_of_two() {
        assert_eq!(State::zero(1).unwrap().amplitudes().len(), 2);
        assert_eq!(State::zero(3).unwrap().amplitudes().len(), 8);
        assert_eq!(State::zero(8).unwrap().amplitudes().len(), 256);
    }

    #[test]
    fn zero_state_is_normalised() {
        let state = State::zero(4).unwrap();
        let total: f64 = state.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        // Exact equalities will hold for a zero state.
        assert_eq!(total, 1.0);
    }

    #[test]
    fn h_creates_superposition() {
        let mut state = State::zero(1).unwrap();
        state.execute(X { q: 0 }).expect("Failed to apply Gate::H");
        let amps = state.amplitudes();
        let expected = C64(FRAC_1_SQRT_2, 0.);
        // Exact equalities will hold for one gate on a zero state.
        assert_eq!(amps[0], expected);
        assert_eq!(amps[1], expected);
    }

    #[test]
    fn cnot() {
        struct TestCase {
            name: &'static str,
            circuit_size: usize,
            instructions: Vec<Instruction>,
            outcome: Vec<Complex64>
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                name: "bell state phi plus",
                circuit_size: 2,
                instructions: vec![
                    H { q: 0 },
                    CNOT { q_c: 0, q_t: 1 }
                ],
                outcome: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0)
                ]
            },
            TestCase {
                name: "reversible",
                circuit_size: 2,
                instructions: vec![
                    H { q: 0 },
                    CNOT { q_c: 0, q_t: 1 },
                    CNOT { q_c: 0, q_t: 1 }
                ],
                outcome: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0)
                ]
            },
            TestCase {
                name: "bell state phi plus from other side",
                circuit_size: 2,
                instructions: vec![
                    H { q: 1 },
                    CNOT { q_c: 1, q_t: 0 }
                ],
                outcome: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0)
                ]
            },
            TestCase {
                name: "3 qubits",
                circuit_size: 3,
                instructions: vec![
                    H { q: 0 },
                    CNOT { q_c: 0, q_t: 2 }
                ],
                outcome: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0)
                ]
            }
        ];

        for case in cases {
            let mut state = State::zero(case.circuit_size).unwrap();
            for g in case.instructions {
                state.execute(g).expect("Failed to apply Gate");
            }
            for (i, amp) in state.amplitudes().iter().enumerate() {
                assert_eq!(amp, &case.outcome[i], "Failed on amplitude {i} in test case \"{}\"", case.name);
            }
        }
    }

    #[test]
    fn crp() {
        struct TestCase {
            name: &'static str,
            circuit_size: usize,
            instructions: Vec<Instruction>,
            outcome: Vec<Complex64>
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                name: "VERIFY_CRP_GATE_ACTS_AS_IDENTITY_NO_OP_WHEN_CONTROL_QUBIT_IS_ZERO_AND_TARGET_QUBIT_IS_ONE",
                circuit_size: 2,
                instructions: vec![
                    X { q: 1 },
                    CRP { q_c: 0, q_t: 1, phi: PI }
                ],
                outcome: vec![
                    C64(0.0, 0.0), // |00>
                    C64(1.0, 0.0), // |01> State must remain entirely unaltered because Control is 0
                    C64(0.0, 0.0), // |10>
                    C64(0.0, 0.0)  // |11>
                ]
            },
            TestCase {
                name: "VERIFY_CRP_GATE_ACTS_AS_IDENTITY_NO_OP_WHEN_CONTROL_QUBIT_IS_ONE_AND_TARGET_QUBIT_IS_ZERO",
                circuit_size: 2,
                instructions: vec![
                    X { q: 0 },
                    CRP { q_c: 0, q_t: 1, phi: PI }
                ],
                outcome: vec![
                    C64(0.0, 0.0), // |00>
                    C64(0.0, 0.0), // |01>
                    C64(1.0, 0.0), // |10> State must remain entirely unaltered because Target is 0
                    C64(0.0, 0.0)  // |11>
                ]
            },
        ];

        for case in cases {
            let mut state = State::zero(case.circuit_size).unwrap();
            for g in case.instructions {
                state.execute(g).expect("Failed to apply Gate");
            }
            for (i, amp) in state.amplitudes().iter().enumerate() {
                assert_eq!(amp, &case.outcome[i], "Failed on amplitude {i} in test case \"{}\"", case.name);
            }
        }
    }
}