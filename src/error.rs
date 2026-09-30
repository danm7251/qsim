use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SimError {
    #[error("qubit index is out of bounds")]
    InvalidQubit,

    #[error("instruction is unsupported")]
    UnsupportedInstruction,

    #[error("a state must contain at least one qubit")]
    ZeroQubits,

    #[error("host CPU does not support AVX")]
    AvxUnsupported,

    #[error("host CPU does not support FMA")]
    FmaUnsupported,

    #[error("config is unsupported")]
    UnsupportedConfig,
}
