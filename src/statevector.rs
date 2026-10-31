use num_complex::Complex64;
use rand::random;

use crate::{
    api::Instruction::{self, *},
    error::SimError,
    kernels,
    linalg::{SquareMatrix, Vector, matrix},
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
pub struct Statevector {
    amplitudes: Vector,
    n: usize,
    config: Config,
}

impl Statevector {
    // Constructors

    /// Creates the `|0...0⟩` state for `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_qubits` is `0`.
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Zero State Construction", err)
    )]
    pub fn zero(num_qubits: usize) -> Result<Self, SimError> {
        // Validation.
        if num_qubits == 0 {
            return Err(SimError::ZeroQubits);
        }

        // Amplitudes setup.
        let mut amplitudes = Vector::zeros(1 << num_qubits);
        *amplitudes.get_mut(0) = Complex64::ONE;

        // Auto Config setup.
        let config = Config {
            avx: is_x86_feature_detected!("avx"),
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
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "Zero State Construction", err)
    )]
    pub fn zero_with_config(num_qubits: usize, config: Config) -> Result<Self, SimError> {
        // Validation.
        if num_qubits == 0 {
            return Err(SimError::ZeroQubits);
        }
        if config.avx && !is_x86_feature_detected!("avx") {
            return Err(SimError::AvxUnsupported);
        }
        if config.fma && !is_x86_feature_detected!("fma") {
            return Err(SimError::FmaUnsupported);
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
    pub fn probabilities(&self, target: usize) -> Result<(f64, f64), SimError> {
        let num_q = self.num_qubits();

        if target >= num_q {
            return Err(SimError::InvalidQubit { index: target, num_qubits: num_q });
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
    pub fn execute(&mut self, cmd: Instruction) -> Result<(), SimError> {
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
            SWAP { q_1, q_2 } => self.apply_swap(q_1, q_2),

            // Subroutines
            QFT => return Err(SimError::UnsupportedInstruction),
        }
    }

    /// Applies an collection of [`Instruction`]s to the state, updating its amplitudes as required.
    ///
    /// Returns an error if an instruction references an invalid qubit or otherwise
    /// cannot be applied to the state.
    pub fn execute_all(&mut self, circuit: &[Instruction]) -> Result<(), SimError> {
        for &cmd in circuit {
            self.execute(cmd)?;
        }

        Ok(())
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
    fn apply_1q(&mut self, target: usize, matrix: &SquareMatrix) -> Result<(), SimError> {
        let num_q = self.n;

        if target >= num_q {
            return Err(SimError::InvalidQubit { index: target, num_qubits: num_q });
        }

        // Convert the target qubit into its state-vector stride.
        let stride = 1 << (num_q - target - 1);

        let config = self.config;
        let amplitudes = self.amplitudes.as_mut_slice();

        // Dispatch to the configured kernel.
        match (config.avx, config.fma) {
            (false, false) => kernels::portable::apply_1q_strided(amplitudes, stride, matrix),
            (false, true) => unsafe { kernels::fma::apply_1q(amplitudes, stride, matrix) },
            (true, false) => {
                if stride > 1 {
                    unsafe { kernels::avx::apply_1q(amplitudes, stride, matrix) }
                } else {
                    kernels::portable::apply_1q_strided(amplitudes, stride, matrix);
                }
            }
            (true, true) => {
                if stride > 1 {
                    unsafe { kernels::avx_fma::apply_1q(amplitudes, stride, matrix) }
                } else {
                    kernels::portable::apply_1q_strided(amplitudes, stride, matrix);
                }
            }
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
    fn apply_c2q(&mut self, control: usize, target: usize, matrix: &SquareMatrix) -> Result<(), SimError> {
        if control >= self.n || target >= self.n {
            return Err(SimError::InvalidQubit { index: target, num_qubits: self.n });
        }
        if control == target {
            return Err(SimError::InvalidQubit { index: target, num_qubits: self.n });
        }

        // Convert the control and target qubits into state-vector strides
        let c_stride = 1 << (self.n - control - 1);
        let t_stride = 1 << (self.n - target - 1);

        let config = self.config;
        let amplitudes = self.amplitudes.as_mut_slice();

        match (config.avx, config.fma) {
            (false, false) => {
                kernels::portable::apply_c2q_strided(amplitudes, c_stride, t_stride, matrix);
            }
            // SAFETY: FMA support was verified when the config was constructed.
            (false, true) => unsafe {
                kernels::fma::apply_c2q(amplitudes, c_stride, t_stride, matrix);
            },
            // SAFETY: AVX support was verified when the config was constructed.
            (true, false) => {
                if c_stride == 1 || t_stride == 1 {
                    kernels::portable::apply_c2q_strided(amplitudes, c_stride, t_stride, matrix);
                } else {
                    unsafe {
                        kernels::avx::apply_c2q(amplitudes, c_stride, t_stride, matrix);
                    }
                }
            },
            // SAFETY: AVX & FMA support were verified when the config was constructed.
            (true, true) => {
                if c_stride == 1 || t_stride == 1 {
                    unsafe { 
                        kernels::fma::apply_c2q(amplitudes, c_stride, t_stride, matrix);
                    }
                } else {
                    unsafe {
                        kernels::avx_fma::apply_c2q(amplitudes, c_stride, t_stride, matrix);
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies a SWAP operation between two qubits.
    ///
    /// The operation exchanges the states of `target_1` and `target_2`.
    ///
    /// # Errors
    ///
    /// Returns [`SimError::InvalidQubit`] if either target qubit does not exist.
    fn apply_swap(&mut self, target_1: usize, target_2: usize) -> Result<(), SimError> {
        if target_1 >= self.n {
            return Err(SimError::InvalidQubit { index: target_1, num_qubits: self.n })
        }

        if target_2 >= self.n {
            return Err(SimError::InvalidQubit { index: target_2, num_qubits: self.n })
        }

        // Swapping a qubit with itself leaves the state unchanged.
        if target_1 == target_2 {
            return Ok(())
        }


        // Decompose SWAP into three CNOT gates.
        let subcircuit = [
            CNOT { q_c: target_1, q_t: target_2 },
            CNOT { q_c: target_2, q_t: target_1 },
            CNOT { q_c: target_1, q_t: target_2 },
        ];

        self.execute_all(&subcircuit)
    }

    // Operations

    /// Measures `target` in the computational basis and collapses the state to
    /// the resulting measurement outcome.
    ///
    /// Returns `true` if qubit is `|1⟩`.
    pub fn measure(&mut self, target: usize) -> Result<bool, SimError> {
        if target >= self.n {
            return Err(SimError::InvalidQubit { index: target, num_qubits: self.n });
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
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    use crate::linalg::C64;

    use super::*;

    /// A test case describing a circuit of instructions and
    /// the expected state amplitudes after applying that circuit.
    struct CircuitTestCase {
        name: &'static str,
        num_qubits: usize,
        instructions: Vec<Instruction>,
        expected_state: Vec<Complex64>,
    }

    /// Executes each [`CircuitCircuitTestCase`]'s circuit on a statevector and
    /// compares the resulting amplitudes against the expected outcome.
    fn run_circuit_test_table(cases: &[CircuitTestCase]) {
        for case in cases {
            let mut state = Statevector::zero(case.num_qubits).unwrap_or_else(|e| {
                panic!("Failed to construct statevector with {} qubits in '{}': {e:?}", case.num_qubits, case.name)
            });

            assert_eq!(
                state.amplitudes().len(),
                case.expected_state.len(),
                "Bad test case: Amplitude count mismatch in '{}'",
                case.name
            );

            for instruction in &case.instructions {
                state.execute(*instruction).unwrap_or_else(|e| {
                    panic!("Failed to execute instruction {instruction:?} in '{}': {e:?}", case.name)
                });
            }

            for (i, amp) in state.amplitudes().iter().enumerate() {
                assert_eq!(
                    amp, &case.expected_state[i],
                    "Failed on amplitude {i} in test case '{}'",
                    case.name
                );
            }
        }
    }

    #[test]
    fn strided_x() {
        let mut state = Statevector::zero(2).unwrap();
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
        assert_eq!(Statevector::zero(1).unwrap().amplitudes().len(), 2);
        assert_eq!(Statevector::zero(3).unwrap().amplitudes().len(), 8);
        assert_eq!(Statevector::zero(8).unwrap().amplitudes().len(), 256);
    }

    #[test]
    fn zero_state_is_normalised() {
        let state = Statevector::zero(4).unwrap();
        let total: f64 = state.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        // Exact equalities will hold for a zero state.
        assert_eq!(total, 1.0);
    }

    #[test]
    fn h_creates_superposition() {
        let mut state = Statevector::zero(1).unwrap();
        state.execute(H { q: 0 }).expect("Failed to apply Gate::H");
        let amps = state.amplitudes();
        let expected = C64(FRAC_1_SQRT_2, 0.);
        // Exact equalities will hold for one gate on a zero state.
        assert_eq!(amps[0], expected);
        assert_eq!(amps[1], expected);
    }

    #[test]
    fn cnot() {
        let cases: Vec<CircuitTestCase> = vec![
            CircuitTestCase {
                name: "bell state phi plus",
                num_qubits: 2,
                instructions: vec![H { q: 0 }, CNOT { q_c: 0, q_t: 1 }],
                expected_state: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                ],
            },
            CircuitTestCase {
                name: "reversible",
                num_qubits: 2,
                instructions: vec![H { q: 0 }, CNOT { q_c: 0, q_t: 1 }, CNOT { q_c: 0, q_t: 1 }],
                expected_state: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                ],
            },
            CircuitTestCase {
                name: "bell state phi plus from other side",
                num_qubits: 2,
                instructions: vec![H { q: 1 }, CNOT { q_c: 1, q_t: 0 }],
                expected_state: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                ],
            },
            CircuitTestCase {
                name: "3 qubits",
                num_qubits: 3,
                instructions: vec![H { q: 0 }, CNOT { q_c: 0, q_t: 2 }],
                expected_state: vec![
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                    C64(FRAC_1_SQRT_2, 0.0),
                    C64(0.0, 0.0),
                    C64(0.0, 0.0),
                ],
            },
        ];

        run_circuit_test_table(&cases);
    }

    #[test]
    fn crp() {
        let cases: Vec<CircuitTestCase> = vec![
            CircuitTestCase {
                name: "VERIFY_CRP_GATE_ACTS_AS_IDENTITY_NO_OP_WHEN_CONTROL_QUBIT_IS_ZERO_AND_TARGET_QUBIT_IS_ONE",
                num_qubits: 2,
                instructions: vec![
                    X { q: 1 },
                    CRP {
                        q_c: 0,
                        q_t: 1,
                        phi: PI,
                    },
                ],
                expected_state: vec![
                    C64(0.0, 0.0), // |00>
                    C64(1.0, 0.0), // |01> State must remain entirely unaltered because Control is 0
                    C64(0.0, 0.0), // |10>
                    C64(0.0, 0.0), // |11>
                ],
            },
            CircuitTestCase {
                name: "VERIFY_CRP_GATE_ACTS_AS_IDENTITY_NO_OP_WHEN_CONTROL_QUBIT_IS_ONE_AND_TARGET_QUBIT_IS_ZERO",
                num_qubits: 2,
                instructions: vec![
                    X { q: 0 },
                    CRP {
                        q_c: 0,
                        q_t: 1,
                        phi: PI,
                    },
                ],
                expected_state: vec![
                    C64(0.0, 0.0), // |00>
                    C64(0.0, 0.0), // |01>
                    C64(1.0, 0.0), // |10> State must remain entirely unaltered because Target is 0
                    C64(0.0, 0.0), // |11>
                ],
            },
        ];

        run_circuit_test_table(&cases);
    }
}
