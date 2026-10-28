use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SimError {
    /// The statevector was initialised with zero qubits.
    #[error("a state must contain at least one qubit")]
    ZeroQubits,

    /// The configuration requires AVX, which the host does not support.
    #[error("host CPU does not support AVX")]
    AvxUnsupported,

    /// The configuration requires FMA, which the host does not support.
    #[error("host CPU does not support FMA")]
    FmaUnsupported,

    /// The specified qubit does not exist in the statevector.
    #[error("qubit index {index} is out of bounds for a {num_qubits}-qubit state")]
    InvalidQubit {
        index: usize,
        num_qubits: usize,
    },

    /// The instruction is not yet implemented.
    ///
    /// TODO: Remove before alpha release by implementing or removing the
    /// corresponding `Instruction` variants.
    #[error("instruction is unsupported")]
    UnsupportedInstruction,

    /// The configuration is not yet implemented.
    /// 
    /// TODO: Remove before alpha release by implementing or removing the
    /// corresponding configuration options.
    #[error("config is unsupported")]
    UnsupportedConfig,
}
