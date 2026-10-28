use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimError {
    #[error("a state must contain at least one qubit")]
    ZeroQubits,

    #[error("host CPU does not support AVX")]
    AvxUnsupported,

    #[error("host CPU does not support FMA")]
    FmaUnsupported,

    #[error("qubit index {index} is out of bounds for a {num_qubits}-qubit state")]
    InvalidQubit {
        index: usize,
        num_qubits: usize,
    },

    #[error("instruction is unsupported")]
    UnsupportedInstruction,

    #[error("config is unsupported")]
    UnsupportedConfig,
}
