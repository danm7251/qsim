#![feature(portable_simd)]

#[cfg(not(target_arch = "x86_64"))]
compile_error!("Only x86_64 is supported by qsim currently");

pub mod api;
pub mod error;
pub mod kernels;
pub mod legacy;
pub mod linalg;
pub mod stabilizer;
pub mod state;