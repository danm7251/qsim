#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    // One Qubit Gates
    X { q: usize },
    Y { q: usize },
    Z { q: usize },
    H { q: usize },
    S { q: usize },
    T { q: usize },
    P { q: usize, phi: f64 },

    // Controlled One Qubit Gates
    CNOT { q_c: usize, q_t: usize },
    CRP { q_c: usize, q_t: usize, phi: f64 },

    // Two Qubit Gates
    SWAP { q_1: usize, q_2: usize },

    // Subroutines
    QFT
}