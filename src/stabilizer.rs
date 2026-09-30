use crate::{api::Instruction, error::SimError};

/// A quantum state represented by a stabilizer tableau.
///
/// The tableau contains 2n rows. The first n rows are destabilizers
/// and the final n rows are stabilizer generators. The destabilizers
/// are not used by any current operation and are retained so that
/// measurement can be added later.
pub struct Stabilizer {
    x: Vec<bool>,
    z: Vec<bool>,
    phase: Vec<bool>,
    n: usize,
}

impl Stabilizer {

    // Constructors

    /// Creates the `|0...0⟩` state for `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_qubits` is `0`.
    pub fn zero(num_qubits: usize) -> Result<Self, SimError> {
        // Validation.
        if num_qubits == 0 {
            return Err(SimError::ZeroQubits);
        }

        // Tableau setup.
        let rows = 2 * num_qubits;
        let mut x = vec![false; rows * num_qubits];
        let mut z = vec![false; rows * num_qubits];
        let phase = vec![false; rows];

        // Destabilizers.
        for qubit in 0..num_qubits {
            x[qubit * num_qubits + qubit] = true;
        }

        // Stabilizers.
        for qubit in 0..num_qubits {
            let row = num_qubits + qubit;
            z[row * num_qubits + qubit] = true;
        }

        Ok(Self {
            x,
            z,
            phase,
            n: num_qubits,
        })
    }

    // Accessors

    /// Returns the number of qubits in the state.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    // Operations

    /// Applies an [`Instruction`] to the stabilizer state.
    ///
    /// Returns an error if the instruction references an invalid qubit or
    /// otherwise cannot be applied to the state.
    pub fn execute(&mut self, cmd: Instruction) -> Result<(), SimError> {
        match cmd {
            // One Qubit Gates

            Instruction::X { q } => self.apply_x(q),
            Instruction::Y { q } => self.apply_y(q),
            Instruction::Z { q } => self.apply_z(q),
            Instruction::H { q } => self.apply_h(q),

            Instruction::S { .. }
            | Instruction::T { .. }
            | Instruction::P { .. } => return Err(SimError::UnsupportedInstruction),

            // Controlled One Qubit Gates

            Instruction::CNOT { q_c, q_t } => self.apply_cnot(q_c, q_t),
            Instruction::CRP { .. } => return Err(SimError::UnsupportedInstruction),

            // Two Qubit Gates

            Instruction::SWAP { .. } => return Err(SimError::UnsupportedInstruction),

            // Subroutines

            Instruction::QFT => return Err(SimError::UnsupportedInstruction),
        }
    }

    fn apply_x(&mut self, target: usize) -> Result<(), SimError> {
        if target >= self.n {
            return Err(SimError::InvalidQubit);
        }

        for row in 0..2 * self.n {
            let index = self.index(row, target);

            // X anticommutes with Z and Y.
            if self.z[index] {
                self.phase[row] ^= true;
            }
        }

        Ok(())
    }

    fn apply_y(&mut self, target: usize) -> Result<(), SimError> {
        if target >= self.n {
            return Err(SimError::InvalidQubit);
        }

        for row in 0..2 * self.n {
            let index = self.index(row, target);

            // Y anticommutes with X and Z, but commutes with Y.
            if self.x[index] ^ self.z[index] {
                self.phase[row] ^= true;
            }
        }

        Ok(())
    }

    fn apply_z(&mut self, target: usize) -> Result<(), SimError> {
        if target >= self.n {
            return Err(SimError::InvalidQubit);
        }

        for row in 0..2 * self.n {
            let index = self.index(row, target);

            // Z anticommutes with X and Y.
            if self.x[index] {
                self.phase[row] ^= true;
            }
        }

        Ok(())
    }

    fn apply_h(&mut self, target: usize) -> Result<(), SimError> {
        if target >= self.n {
            return Err(SimError::InvalidQubit);
        }

        for row in 0..2 * self.n {
            let index = self.index(row, target);

            // H maps X <-> Z and Y -> -Y.
            if self.x[index] && self.z[index] {
                self.phase[row] ^= true;
            }

            std::mem::swap(&mut self.x[index], &mut self.z[index]);
        }

        Ok(())
    }

    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        if control >= self.n || target >= self.n {
            return Err(SimError::InvalidQubit);
        }

        if control == target {
            return Err(SimError::InvalidQubit);
        }

        for row in 0..2 * self.n {
            let control_index = self.index(row, control);
            let target_index = self.index(row, target);

            // Update the generator phase before modifying the Pauli components.
            if self.x[control_index]
                && self.z[target_index]
                && (self.x[target_index] ^ self.z[control_index] ^ true)
            {
                self.phase[row] ^= true;
            }

            self.x[target_index] ^= self.x[control_index];
            self.z[control_index] ^= self.z[target_index];
        }

        Ok(())
    }

    #[inline]
    fn index(&self, row: usize, qubit: usize) -> usize {
        row * self.n + qubit
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::statevector::Statevector;

    #[test]
    fn zero_state_is_initialised_correctly() {
        let state = Stabilizer::zero(2).unwrap();

        // First two rows are destabilizers: X0, X1.
        assert!(state.x[0]);
        assert!(state.x[3]);

        // Last two rows are stabilizers: Z0, Z1.
        assert!(state.z[4]);
        assert!(state.z[7]);

        // All generators initially have positive phase.
        assert!(state.phase.iter().all(|&phase| !phase));
    }

    #[test]
    fn x_flips_z_stabilizer_phase() {
        let mut state = Stabilizer::zero(1).unwrap();

        state.apply_x(0).unwrap();

        // |0> -> |1>, so Z becomes -Z.
        assert!(state.z[1]);
        assert!(state.phase[1]);
    }

    #[test]
    fn y_flips_x_and_z_stabilizer_phases() {
        let mut state = Stabilizer::zero(1).unwrap();

        state.apply_y(0).unwrap();

        // Y anticommutes with both X and Z.
        assert!(state.x[0]);
        assert!(state.z[1]);
        assert!(state.phase[0]);
        assert!(state.phase[1]);
    }

    #[test]
    fn z_flips_x_stabilizer_phase() {
        let mut state = Stabilizer::zero(1).unwrap();

        state.apply_z(0).unwrap();

        // Z anticommutes with X.
        assert!(state.x[0]);
        assert!(state.phase[0]);

        // Z commutes with Z, so its phase is unchanged.
        assert!(state.z[1]);
        assert!(!state.phase[1]);
    }

    #[test]
    fn pauli_twice_returns_to_original_state() {
        let mut state = Stabilizer::zero(2).unwrap();

        state.apply_x(0).unwrap();
        state.apply_x(0).unwrap();

        state.apply_y(1).unwrap();
        state.apply_y(1).unwrap();

        state.apply_z(0).unwrap();
        state.apply_z(0).unwrap();

        assert!(state.phase.iter().all(|&phase| !phase));
    }

    #[test]
    fn pauli_rejects_invalid_qubit() {
        let mut state = Stabilizer::zero(2).unwrap();

        assert_eq!(state.apply_x(2), Err(SimError::InvalidQubit));
        assert_eq!(state.apply_y(2), Err(SimError::InvalidQubit));
        assert_eq!(state.apply_z(2), Err(SimError::InvalidQubit));
    }

    #[test]
    fn execute_applies_pauli_gates() {
        let mut state = Stabilizer::zero(1).unwrap();

        state.execute(Instruction::X { q: 0 }).unwrap();

        assert!(state.z[1]);
        assert!(state.phase[1]);

        state.execute(Instruction::Z { q: 0 }).unwrap();

        assert!(state.x[0]);
        assert!(state.phase[0]);
    }

    #[test]
    fn cnot_creates_bell_state_stabilizers() {
        let mut state = Stabilizer::zero(2).unwrap();

        state.apply_h(0).unwrap();
        state.apply_cnot(0, 1).unwrap();

        // S0 = X0 X1.
        assert!(state.x[4]);
        assert!(state.x[5]);
        assert!(!state.z[4]);
        assert!(!state.z[5]);

        // S1 = Z0 Z1.
        assert!(!state.x[6]);
        assert!(!state.x[7]);
        assert!(state.z[6]);
        assert!(state.z[7]);

        // No generator has acquired a negative phase.
        assert!(state.phase.iter().all(|&phase| !phase));
    }

    #[test]
    fn h_and_cnot_reject_invalid_qubits() {
        let mut state = Stabilizer::zero(2).unwrap();

        assert_eq!(state.apply_h(2), Err(SimError::InvalidQubit));
        assert_eq!(state.apply_cnot(2, 0), Err(SimError::InvalidQubit));
        assert_eq!(state.apply_cnot(0, 2), Err(SimError::InvalidQubit));
        assert_eq!(state.apply_cnot(0, 0), Err(SimError::InvalidQubit));
    }

    // Statevector cross-check.
    //
    // Every stabilizer generator, including its sign, must leave the
    // state produced by the statevector simulator unchanged.

    const EPSILON: f64 = 1e-14;

    /// Every gate the stabilizer simulator supports, on `n` qubits.
    fn supported_gates(n: usize) -> Vec<Instruction> {
        let mut gates = Vec::new();

        for q in 0..n {
            gates.push(Instruction::X { q });
            gates.push(Instruction::Y { q });
            gates.push(Instruction::Z { q });
            gates.push(Instruction::H { q });
        }

        for q_c in 0..n {
            for q_t in 0..n {
                if q_c != q_t {
                    gates.push(Instruction::CNOT { q_c, q_t });
                }
            }
        }

        gates
    }

    fn assert_stabilizes(n: usize, circuit: &[Instruction]) {
        let mut tableau = Stabilizer::zero(n).unwrap();

        for &cmd in circuit {
            tableau.execute(cmd).unwrap();
        }

        let mut reference = Statevector::zero(n).unwrap();
        reference.execute_all(circuit).unwrap();
        let expected = reference.amplitudes().to_vec();

        // Only the stabilizer generators are checked.
        for row in n..2 * n {
            let mut state = Statevector::zero(n).unwrap();
            state.execute_all(circuit).unwrap();

            // Apply the Pauli operator described by the row, ignoring its sign.
            for q in 0..n {
                let index = tableau.index(row, q);

                match (tableau.x[index], tableau.z[index]) {
                    (true, false) => state.execute(Instruction::X { q }).unwrap(),
                    (false, true) => state.execute(Instruction::Z { q }).unwrap(),
                    (true, true) => state.execute(Instruction::Y { q }).unwrap(),
                    (false, false) => {}
                }
            }

            // The state must be unchanged for a positive phase and negated
            // for a negative phase.
            let sign = if tableau.phase[row] { -1.0 } else { 1.0 };

            for (actual, expected) in state.amplitudes().iter().zip(&expected) {
                assert!(
                    (*actual - *expected * sign).norm() < EPSILON,
                    "Generator {row} does not stabilise the state for circuit {circuit:?}"
                );
            }
        }
    }

    #[test]
    fn generators_stabilise_statevector_for_all_short_circuits() {
        const QUBITS: usize = 3;
        const MAX_LENGTH: u32 = 3;

        let gates = supported_gates(QUBITS);

        for length in 0..=MAX_LENGTH {
            let total = gates.len().pow(length);

            for mut code in 0..total {
                let mut circuit = Vec::new();

                for _ in 0..length {
                    circuit.push(gates[code % gates.len()]);
                    code /= gates.len();
                }

                assert_stabilizes(QUBITS, &circuit);
            }
        }
    }
}