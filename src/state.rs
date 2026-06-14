use ndarray::{array, Array1, Array2};
use num_complex::{Complex64, ComplexFloat};
use rand::random;

use crate::{gates::Gate, math_utils::{C64, kron}};

pub struct State {
    amplitudes: Array1<Complex64>, // Big endian, for now.
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
    pub fn zero(circuit_size: usize) -> Result<Self, &'static str> {
        if circuit_size == 0 {
            // I will want to implement a custom error type soon.
            return Err("A state with 0 qubits is invalid");
        }

        let mut amplitudes = Array1::zeros(1 << circuit_size);
        amplitudes[0] = Complex64::ONE;

        Ok(Self {
            amplitudes,
            n: circuit_size
        })
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

    // Functionality
    /// Applies a 'Gate' instruction to the state.
    pub fn apply_gate(&mut self, gate: Gate) -> Result<(), &str> {
        match gate {
            // Consider that while sharing functionality, because CRP has to carry a parameterised matrix,
            // it has a different structure to CNOT. This is inconvenient.
            Gate::CNOT { control, target } => self.apply_c2q_index(control, target, gate.matrix()),
            Gate::CRP { control, target, .. } => self.apply_c2q_index(control, target, gate.matrix()),
            // Another great reason for switching to a public facing Instruction enum,
            // I use the Identity matrix internally but there is no reason to expose this to users.
            Gate::I => Err("Why would you use this"),
            // It would be nice if I could handle these all in the same case but,
            // I do not want to bloat the API with useless distinctions,
            // for example, Gate::UnparamOneQubit { Matrix::X } = bad.
            Gate::X { target } => self.apply_1q_index(target, gate.matrix()),
            Gate::Y { target } => self.apply_1q_index(target, gate.matrix()),
            Gate::Z { target } => self.apply_1q_index(target, gate.matrix()),
            Gate::H { target } => self.apply_1q_index(target, gate.matrix()),
            Gate::S { target } => self.apply_1q_index(target, gate.matrix()),
            Gate::T { target } => self.apply_1q_index(target, gate.matrix())
        }
    }

    fn apply_1q_index(&mut self, target: usize, gate_matrix: Array2<Complex64>) -> Result<(), &str> {
        if target >= self.n {
            return Err("Target qubit does not exist")
        }

        for index_low in 0..self.amplitudes.len() {
            if index_is_zero(self.n, target, index_low) {
                // Find index where the bitstring only differs on `target`.
                let index_high = index_low + (1 << self.n - target - 1);
                // Create a pair/subspace.
                let pair: Array1<Complex64> = array![self.amplitudes[index_low], self.amplitudes[index_high]];
                // Perform operation.
                let updated_pair = gate_matrix.dot(&pair);
                // Replace amplitudes with updated values.
                self.amplitudes[index_low] = updated_pair[0];
                self.amplitudes[index_high] = updated_pair[1];
            }
        }

        Ok(())
    }

    // Controlled two-qubit general
    // While initially I had a specialised CNOT kernel,
    // and often specific implementations of gates are more efficient.
    // Maintaining specialised functions for each gate would be a nightmare.
    fn apply_c2q_index(&mut self, control: usize, target: usize, gate_matrix: Array2<Complex64>) -> Result<(), &str> {
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
                let pair: Array1<Complex64> = array![self.amplitudes[index_low], self.amplitudes[index_high]];
                // Perform operation.
                let updated_pair = gate_matrix.dot(&pair);
                // Replace amplitudes with updated values.
                self.amplitudes[index_low] = updated_pair[0];
                self.amplitudes[index_high] = updated_pair[1];
            }
        }

        Ok(())
    }

    /// Applies a one-qubit gate to a target qubit.
    /// Will be replaced in future as it constructs a full system matrix.
    #[deprecated]
    #[allow(unused)]
    // Not inlined for easy memory profiling
    #[inline(never)]
    fn apply_1q_kron(&mut self, target: usize, gate_matrix: Array2<Complex64>) -> Result<(), &str> {
        if target >= self.n {
            return Err("Target qubit does not exist")
        }

        // Store one base identity matrix to borrow from.
        let identity = Gate::I.matrix();

        // Define what the term at i should be.
        let matrix_fn = |i: usize| {
            if target == i { &gate_matrix }
            else { &identity }
        };

        // Define first term.
        let mut matrix = matrix_fn(0).clone();

        // Chain all other terms.
        for i in 1..self.n {
            matrix = kron(&matrix, matrix_fn(i))
        }

        // Update state.
        self.amplitudes = matrix.dot(&self.amplitudes);
        Ok(())
    }

    /// Applies a two-qubit gate with one control qubit to a target qubit.
    /// Not recommended for use as it constructs a full state matrix.
    #[deprecated]
    #[allow(unused)]
    // Not inlined for easy memory profiling
    #[inline(never)]
    fn apply_c2q_kron(&mut self, control: usize, target: usize, gate_matrix: Array2<Complex64>) -> Result<(), &str> {
        if control == target {
            return Err("Control and target must be distinct qubits")
        } else if control >= self.n || target >= self.n {
            return Err("Control and target must be an existing qubit")
        }

        // Define frequently used matrices.
        let identity = Gate::I.matrix();
        let p0 = array![[C64(1., 0.), C64(0., 0.)], [C64(0., 0.), C64(0., 0.)]];
        let p1 = array![[C64(0., 0.), C64(0., 0.)], [C64(0., 0.), C64(1., 0.)]];

        // Defines what M(0) at i should be.
        let m0_fn = |i: usize| {
            if control == i { &p0 }
            else { &identity }
        };

        // Defines what M(1) at i should be.
        let m1_fn = |i: usize| { 
            if control == i { &p1 }
            else if target == i { &gate_matrix }
            else { &identity }
        };

        // Define first element
        let mut m0 = m0_fn(0).clone();
        let mut m1 = m1_fn(0).clone();

        // Continue to chain kronneckers using closures.
        for i in 1..self.n {
            m0 = kron(&m0, m0_fn(i));
            m1 = kron(&m1, m1_fn(i));
        }

        // Update amplitudes by applying U where U = M(0) + M(1).
        self.amplitudes = (m0 + m1).dot(&self.amplitudes);

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
    use std::f64::consts::{PI, SQRT_2};

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
        state.apply_gate(Gate::H {target: 0}).expect("Failed to apply Gate::H");
        let amps = state.amplitudes();
        let expected = C64(1.0 / SQRT_2, 0.);
        // Exact equalities will hold for one gate on a zero state.
        assert_eq!(amps[0], expected);
        assert_eq!(amps[1], expected);
    }

    #[test]
    fn cnot() {
        struct TestCase {
            name: &'static str,
            circuit_size: usize,
            instructions: Vec<Gate>,
            outcome: Vec<Complex64>
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                name: "bell state phi plus",
                circuit_size: 2,
                instructions: vec![
                    Gate::H { target: 0 },
                    Gate::CNOT { control: 0, target: 1 }
                ],
                outcome: vec![
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(1.0/SQRT_2, 0.0)
                ]
            },
            TestCase {
                name: "reversible",
                circuit_size: 2,
                instructions: vec![
                    Gate::H { target: 0 },
                    Gate::CNOT { control: 0, target: 1 },
                    Gate::CNOT { control: 0, target: 1 }
                ],
                outcome: vec![
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0)
                ]
            },
            TestCase {
                name: "bell state phi plus from other side",
                circuit_size: 2,
                instructions: vec![
                    Gate::H { target: 1 },
                    Gate::CNOT { control: 1, target: 0 }
                ],
                outcome: vec![
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(1.0/SQRT_2, 0.0)
                ]
            },
            TestCase {
                name: "3 qubits",
                circuit_size: 3,
                instructions: vec![
                    Gate::H { target: 0 },
                    Gate::CNOT { control: 0, target: 2 }
                ],
                outcome: vec![
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(1.0/SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0)
                ]
            }
        ];

        for case in cases {
            let mut state = State::zero(case.circuit_size).unwrap();
            for g in case.instructions {
                state.apply_gate(g).expect("Failed to apply Gate");
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
            instructions: Vec<Gate>,
            outcome: Vec<Complex64>
        }

        let cases: Vec<TestCase> = vec![
            TestCase {
                name: "VERIFY_CRP_GATE_ACTS_AS_IDENTITY_NO_OP_WHEN_CONTROL_QUBIT_IS_ZERO_AND_TARGET_QUBIT_IS_ONE",
                circuit_size: 2,
                instructions: vec![
                    Gate::X { target: 1 },
                    Gate::CRP { control: 0, target: 1, phi: PI }
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
                    Gate::X { target: 0 },
                    Gate::CRP { control: 0, target: 1, phi: PI }
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
                state.apply_gate(g).expect("Failed to apply Gate");
            }
            for (i, amp) in state.amplitudes().iter().enumerate() {
                assert_eq!(amp, &case.outcome[i], "Failed on amplitude {i} in test case \"{}\"", case.name);
            }
        }
    }
}