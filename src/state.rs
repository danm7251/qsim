use num_complex::Complex64;
use rand::random;

use crate::{
    api::Instruction::{self, *}, kernels, linalg::{matrix, SquareMatrix, Vector}
};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub avx: bool,
    pub fma: bool,
}

/// A quantum state represented by a statevector.
///
/// Amplitudes use big-endian qubit ordering, with qubit 0 corresponding
/// to the most significant bit of the amplitude index.
pub struct State {
    amplitudes: Vector,
    n: usize,
    config: Config,
}

impl State {
    // Constructors

    /// Creates the `|0...0⟩` state for `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_qubits` is `0`.
    #[cfg_attr(feature = "trace", tracing::instrument(name = "Zero State Construction", err))]
    pub fn zero(num_qubits: usize) -> Result<Self, &'static str> {
        // Validation.
        if num_qubits == 0 {
            return Err("A state with 0 qubits is invalid");
        }

        // Amplitudes setup.
        let mut amplitudes = Vector::zeros(1 << num_qubits);
        *amplitudes.get_mut(0) = Complex64::ONE;

        // Auto Config setup.
        let config = Config {
            avx: false, //is_x86_feature_detected!("avx"),
            fma: is_x86_feature_detected!("fma"),
        };

        Ok(Self {
            amplitudes,
            n: num_qubits,
            config: config,
        })
    }

    /// Creates the `|0...0⟩` state for `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `num_qubits` is `0`.
    /// - host CPU does not support `config` values.
    #[cfg_attr(feature = "trace", tracing::instrument(name = "Zero State Construction", err))]
    pub fn zero_with_config(num_qubits: usize, config: Config) -> Result<Self, &'static str> {
        // Validation.
        if num_qubits == 0 {
            return Err("A state with 0 qubits is invalid");
        }
        if config.avx && !is_x86_feature_detected!("avx") {
            return Err("Host CPU does not support AVX");
        }
        if config.fma && !is_x86_feature_detected!("fma") {
            return Err("Host CPU does not support FMA");
        }

        // Amplitudes setup.
        let mut amplitudes = Vector::zeros(1 << num_qubits);
        *amplitudes.get_mut(0) = Complex64::ONE;

        Ok(Self {
            amplitudes,
            n: num_qubits,
            config: config,
        })
    }

    // Accessors

    /// Returns a slice containing the state's probability amplitudes.
    pub fn amplitudes(&self) -> &[Complex64] {
        self.amplitudes.as_slice()
    }

    /// Returns the number of qubits in the state.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Returns the squared L2 norm of the state.
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .as_slice()
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum()
    }

    /// Returns the marginal probability distribution of the target qubit.
    ///
    /// The returned tuple contains the probabilities of measuring the qubit as
    /// `|0⟩` and `|1⟩`, respectively.
    pub fn probabilities(&self, target: usize) -> Result<(f64, f64), &'static str> {
        let num_q = self.num_qubits();

        if target >= num_q {
            return Err("Target qubit does not exist");
        }

        let stride = 1 << (num_q - target - 1);
        let mut prob_0 = 0.0;

        for offset in (0..self.amplitudes.len()).step_by(2 * stride) {
            for i in offset..(offset + stride) {
                prob_0 += self.amplitudes.get(i).norm_sqr();
            }
        }

        let prob_1 = 1.0 - prob_0;

        Ok((prob_0, prob_1))
    }

    // Instruction routing

    /// Applies an [`Instruction`] to the state, updating its amplitudes as required.
    ///
    /// Returns an error if the instruction references an invalid qubit or otherwise
    /// cannot be applied to the state.
    pub fn execute(&mut self, cmd: Instruction) -> Result<(), &'static str> {
        match cmd {
            // One Qubit Gates
            X { q } => self.apply_1q(q, &matrix::x()),
            Y { q } => self.apply_1q(q, &matrix::y()),
            Z { q } => self.apply_1q(q, &matrix::z()),
            H { q } => self.apply_1q(q, &matrix::h()),
            S { q } => self.apply_1q(q, &matrix::s()),
            T { q } => self.apply_1q(q, &matrix::t()),
            P { q, phi } => self.apply_1q(q, &matrix::p(phi)),

            // Controlled One Qubit Gates
            CNOT { q_c, q_t } => self.apply_c2q(q_c, q_t, &matrix::x()),
            CRP { q_c, q_t, phi } => self.apply_c2q(q_c, q_t, &matrix::p(phi)),

            // Two Qubit Gates
            SWAP { .. } => unimplemented!("SWAP is unimplemented!"),

            // Subroutines
            QFT => unimplemented!("QFT is unimplemented!"),
        }
    }

    // Gate kernels

    /// Applies a single-qubit `matrix` to `target`.
    ///
    /// Selects the configured kernel after validating the target and calculating
    /// its state-vector stride.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not identify an existing qubit.
    #[cfg_attr(feature = "bench", visibility::make(pub))]
    fn apply_1q(&mut self, target: usize, matrix: &SquareMatrix) -> Result<(), &'static str> {
        let num_q = self.n;

        if target >= num_q {
            return Err("Target qubit does not exist");
        }

        // Convert the target qubit into its state-vector stride.
        let stride = 1 << (num_q - target - 1);
        
        let config = self.config;
        let amplitudes = self.amplitudes.as_mut_slice();

        // Dispatch to the configured kernel.
        match (config.avx, config.fma) {
            (false, false) => kernels::generic::apply_1q(amplitudes, stride, matrix),
            (false, true) => unsafe { kernels::fma::apply_1q(amplitudes, stride, matrix) },
            (true, false) => if stride > 1 {
                unsafe { kernels::avx::apply_1q(amplitudes, stride, matrix) }
            } else {
                kernels::generic::apply_1q(amplitudes, stride, matrix);
            },
            _ => unimplemented!("AVX and FMA are unimplemented!"),
        }

        Ok(())
    }

    /// Applies a controlled single-qubit `matrix` to `target`.
    ///
    /// The operation is applied only to amplitudes where `control` is `|1⟩`.
    /// After validation, both qubits are converted into state-vector strides and
    /// passed to the configured kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if either qubit does not exist or if `control` and `target`
    /// identify the same qubit.
    #[cfg_attr(feature = "bench", visibility::make(pub))]
    fn apply_c2q(&mut self, control: usize, target: usize, matrix: &SquareMatrix) -> Result<(), &'static str> {
        if control >= self.n || target >= self.n {
            return Err("Control and target must be existing qubits");
        }
        if control == target {
            return Err("Control and target must be distinct qubits");
        }

        // Convert the control and target qubits into state-vector strides
        let c_stride = 1 << (self.n - control - 1);
        let t_stride = 1 << (self.n - target - 1);

        let config = self.config;
        let amplitudes = self.amplitudes.as_mut_slice();

        match (config.avx, config.fma) {
            (false, false) => kernels::generic::apply_c2q(amplitudes, c_stride, t_stride, matrix),
            (false, true) => unsafe { kernels::fma::apply_c2q(amplitudes, c_stride, t_stride, matrix) },
            _ => unimplemented!("AVX and FMA are unimplemented!"),
        }

        Ok(())
    }

    // Operations

    /// Measures `target` in the computational basis and collapses the state to
    /// the resulting measurement outcome.
    /// 
    /// Returns `true` if qubit is `|1⟩`.
    pub fn measure(&mut self, target: usize) -> Result<bool, &'static str> {
        if target >= self.n {
            return Err("Target qubit does not exist");
        }

        let (prob_0, prob_1) = self.probabilities(target)?;

        // Sample measurement outcome.
        let outcome_is_one = random::<f64>() > prob_0;

        #[cfg(feature = "trace")]
        tracing::debug!(
            "Measurement: target={}, P(0)={}, P(1)={}, outcome={}",
            target,
            prob_0,
            prob_1,
            outcome_is_one as u8,
        );

        let stride = 1 << (self.n - target - 1);

        // Collapse and renormalise the state.
        for offset in (0..self.amplitudes.len()).step_by(2 * stride) {
            for index_low in offset..(offset + stride) {
                let index_high = index_low + stride;

                // Renormalise amplitudes matching the measurement outcome and zero the rest.
                if outcome_is_one {
                    *self.amplitudes.get_mut(index_low) = Complex64::ZERO;
                    *self.amplitudes.get_mut(index_high) /= prob_1.sqrt();
                } else {
                    *self.amplitudes.get_mut(index_low) /= prob_0.sqrt();
                    *self.amplitudes.get_mut(index_high) = Complex64::ZERO;
                }
            }
        }

        Ok(outcome_is_one)
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::{PI, FRAC_1_SQRT_2};

    use crate::legacy::math_utils::C64;

    use super::*;

    #[test]
    fn strided_x() {
        let mut state = State::zero(2).unwrap();
        state.apply_1q(0, &matrix::h()).unwrap();
        
        let amps = state.amplitudes();
        let expected = Vector::from_elements([
            C64(FRAC_1_SQRT_2, 0.0),
            C64(0.0, 0.0),
            C64(FRAC_1_SQRT_2, 0.0),
            C64(0.0, 0.0),
        ]);

        for i in 0..expected.len() {
            assert_eq!(amps[i], *expected.get(i));
        }
    }

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
        state.execute(H { q: 0 }).expect("Failed to apply Gate::H");
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