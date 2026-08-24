//! Generic state-vector gate kernels.
//!
//! These kernels operate on validated amplitude slices and qubit strides
//! provided by [`State`](crate::state::State).

use num_complex::Complex64;

use crate::linalg::{linear_map, SquareMatrix, Vector};

/// Applies `matrix` to every amplitude pair separated by `t_stride`.
///
/// `t_stride` identifies the target qubit's stride within `amps`.
///
/// # Panics
///
/// Panics if `t_stride` is zero or does not describe a valid partition of
/// `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "1 Qubit Gate Strided"))]
pub fn apply_1q(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) -> () {
    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in offset..(offset + t_stride) {
            apply_pair(amps, index_low, t_stride, matrix);
        }
    }
}

/// Applies `matrix` to amplitude pairs where the control qubit is `|1⟩`.
///
/// `c_stride` and `t_stride` identify the control and target qubits'
/// respective strides within `amps`.
///
/// # Panics
///
/// Panics if either stride is zero, the strides are equal, or they do not
/// describe valid qubits within `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "2 Qubit Gate Strided"))]
pub fn apply_c2q(amps: &mut[Complex64], c_stride: usize, t_stride: usize, matrix: &SquareMatrix) -> () {
    if c_stride < t_stride {
        // Target is more significant, so select T=0 blocks before C=1 blocks.
        for t_block in (0..amps.len()).step_by(2 * t_stride) {
            let t_is_zero = t_block..(t_block + t_stride);

            for c_block in t_is_zero.step_by(2 * c_stride) {
                let c_is_one = (c_block + c_stride)..(c_block + 2 * c_stride);

                for index_low in c_is_one {
                    apply_pair(amps, index_low, t_stride, matrix);
                }
            }
        }
    } else {
        // Control is more significant, so select C=1 blocks before T=0 blocks.
        for c_block in (c_stride..amps.len()).step_by(2 * c_stride) {
            let c_is_one = c_block..(c_block + c_stride);

            for t_block in c_is_one.step_by(2 * t_stride) {
                let t_is_zero = t_block..(t_block + t_stride);

                for index_low in t_is_zero {
                    apply_pair(amps, index_low, t_stride, matrix);
                }
            }
        }
    }
}

/// Applies `matrix` to the amplitude pair beginning at `index_low`.
///
/// The paired amplitude is located at `index_low + t_stride`.
///
/// # Panics
///
/// Panics if either amplitude index is outside `amps`.
fn apply_pair(amps: &mut[Complex64], index_low: usize, t_stride: usize, matrix: &SquareMatrix) {
    let index_high = index_low + t_stride;
    let input = [amps[index_low], amps[index_high]];
    let mut output = [Complex64::ZERO; 2];

    for i in 0..2 {
        let mut total = Complex64::ZERO;

        for j in 0..2 {
            let coefficient = *matrix.get(i, j);
            let value = input[j];

            total = Complex64::new(
                coefficient.re * value.re + (-coefficient.im * value.im + total.re),
                coefficient.re * value.im + (coefficient.im * value.re + total.im),
            );
        }

        output[i] = total;
    }

    amps[index_low] = output[0];
    amps[index_high] = output[1];
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::matrix;

    // Floating point error tolerance.
    const EPSILON: f64 = 1e-12;

    fn assert_amps_eq(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());

        for (index, (actual, expected)) in
            actual.iter().zip(expected).enumerate()
        {
            let difference = (*actual - *expected).norm();

            assert!(
                difference < EPSILON,
                "Amplitude {index} differs: {actual} != {expected}"
            );
        }
    }

    fn real_amps(values: &[f64]) -> Vec<Complex64> {
        values
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect()
    }

    #[test]
    fn pair_applies_gate_to_selected_amplitudes() {
        let mut amps = real_amps(&[1.0, 0.0, 0.0, 0.0]);

        apply_pair(&mut amps, 0, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[0.0, 0.0, 1.0, 0.0]),
        );
    }

    #[test]
    fn one_qubit_gate_with_unit_stride() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_1q(&mut amps, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[2.0, 1.0, 4.0, 3.0]),
        );
    }

    #[test]
    fn one_qubit_gate_with_larger_stride() {
        let mut amps =
            real_amps(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        apply_1q(&mut amps, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]),
        );
    }

    #[test]
    fn controlled_gate_with_more_significant_control() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        // Control q0 has stride 2; target q1 has stride 1.
        apply_c2q(&mut amps, 2, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 2.0, 4.0, 3.0]),
        );
    }

    #[test]
    fn controlled_gate_with_more_significant_target() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        // Control q1 has stride 1; target q0 has stride 2.
        apply_c2q(&mut amps, 1, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 4.0, 3.0, 2.0]),
        );
    }

    #[test]
    fn complex_gate_is_applied_correctly() {
        let mut amps = real_amps(&[1.0, 0.0]);

        apply_1q(&mut amps, 1, &matrix::y());

        assert_amps_eq(
            &amps,
            &[
                Complex64::ZERO,
                Complex64::new(0.0, 1.0),
            ],
        );
    }

    #[test]
    fn unitary_gate_preserves_norm() {
        let mut amps = real_amps(&[1.0, 0.0, 0.0, 0.0]);

        apply_1q(&mut amps, 2, &matrix::h());

        let norm: f64 = amps.iter().map(Complex64::norm_sqr).sum();

        assert!((norm - 1.0).abs() < EPSILON);
    }
}