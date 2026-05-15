use ndarray::Array1;
use num_complex::Complex64;

pub struct State {
    amplitudes: Array1<Complex64>,
    circuit_size: usize
}

impl State {
    pub fn zero(circuit_size: usize) -> Self {
        let mut amplitudes = Array1::zeros(1 << circuit_size);
        amplitudes[0] = Complex64::ONE;

        println!("{}", amplitudes.len());

        Self {
            amplitudes,
            circuit_size
        }
    }
}