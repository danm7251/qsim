//! Public types for describing quantum circuit instructions.

/// An instruction that can be applied to a quantum state.
#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    // Single-qubit gates

    /// Pauli-X gate acting on qubit `q`.
    X { q: usize },

    /// Pauli-Y gate acting on qubit `q`.
    Y { q: usize },

    /// Pauli-Z gate acting on qubit `q`.
    Z { q: usize },

    /// Hadamard gate acting on qubit `q`.
    H { q: usize },

    /// Phase gate acting on qubit `q`.
    S { q: usize },

    /// T gate acting on qubit `q`.
    T { q: usize },

    /// Phase-shift gate acting on qubit `q` with phase angle `phi` in radians.
    P { q: usize, phi: f64 },

    // Controlled single-qubit gates

    /// Controlled-NOT gate, with control qubit `q_c` and target qubit `q_t`.
    CNOT { q_c: usize, q_t: usize },

    // Controlled phase-shift gate with control qubit `q_c`, target qubit
    /// `q_t`, and phase angle `phi` in radians.
    CRP { q_c: usize, q_t: usize, phi: f64 },

    // Two-qubit gates

    /// SWAP gate between qubits `q_1` and `q_2`.
    SWAP { q_1: usize, q_2: usize },

    // Subroutines

    /// Quantum Fourier transform over the entire state.
    QFT,
}
