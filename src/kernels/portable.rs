//! Generic state-vector gate kernels.
//!
//! These kernels operate on validated amplitude slices and qubit strides
//! provided by [`State`](crate::state::State). Gate matrices must be 2 × 2,
//! and amplitude slices must have a power-of-two length.

use num_complex::Complex64;

use crate::linalg::{matrix, SquareMatrix};

/// Applies `matrix` to every amplitude pair separated by `t_stride`.
///
/// `t_stride` must identify a valid target-qubit stride within `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "1 Qubit Gate (Strided)"))]
pub fn apply_1q_strided(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) {
    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in offset..(offset + t_stride) {
            apply_pair(amps, index_low, t_stride, matrix);
        }
    }
}

/// Applies `matrix` to amplitude pairs where the control qubit is `|1⟩`.
///
/// `c_stride` and `t_stride` must identify valid control and target qubit
/// strides within `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "2 Qubit Gate (Strided)"))]
pub fn apply_c2q_strided(amps: &mut[Complex64], c_stride: usize, t_stride: usize, matrix: &SquareMatrix) {
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

/// Applies `matrix` by expanding it into the full system matrix.
///
/// `t_stride` must identify a valid target-qubit stride within `amps`.
#[allow(dead_code)]
#[deprecated = "retained for reference, use apply_1q_strided"]
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "1 Qubit Gate (Kronecker)"))]
pub fn apply_1q_kronecker(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) {
    let num_amps = amps.len();

    debug_assert!(
        num_amps >= 2 && num_amps.is_power_of_two(),
        "Number of amplitudes must be a power of two and at least two."
    );

    debug_assert!(
        t_stride.is_power_of_two() && t_stride <= num_amps / 2,
        "Target stride must be a power of two and no larger than half the number of amplitudes."
    );

    debug_assert!(
        matrix.size() == 2,
        "Single-qubit gates must have dimension two."
    );

    let num_qubits = num_amps.ilog2() as usize;
    let target_qubit = num_qubits - 1 - t_stride.ilog2() as usize;

    let identity = matrix::i();

    // Expand the gate across the full system:
    //
    // U = I ⊗ ... ⊗ matrix ⊗ ... ⊗ I
    //
    // `matrix` acts on the target qubit, with identity on every other qubit.

    let matrix_fn = |i: usize| {
        if target_qubit == i { matrix }
        else { &identity }
    };

    let mut system_matrix = matrix_fn(0).clone();

    for i in 1..num_qubits {
        system_matrix = kronecker_product(&system_matrix, matrix_fn(i));
    }

    let mut output = vec![Complex64::ZERO; num_amps];

    for (row, result) in output.iter_mut().enumerate() {
        for (col, input) in amps.iter().enumerate() {
            *result += system_matrix.get(row, col) * input;
        }
    }

    amps.copy_from_slice(&output);
}

/// Applies controlled `matrix` by expanding it into the full system matrix.
///
/// `c_stride` and `t_stride` must identify distinct valid control- and
/// target-qubit strides within `amps`.
#[allow(dead_code)]
#[deprecated = "retained for reference, use apply_c2q_strided"]
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "2 Qubit Gate (Kronecker)"))]
pub fn apply_c2q_kronecker(amps: &mut[Complex64], c_stride: usize, t_stride: usize, matrix: &SquareMatrix) {
    let num_amps = amps.len();

    debug_assert!(
        num_amps >= 4 && num_amps.is_power_of_two(),
        "Number of amplitudes must be a power of two and at least four."
    );

    debug_assert!(
        c_stride.is_power_of_two() && c_stride <= num_amps / 2,
        "Control stride must be a power of two and no larger than half the number of amplitudes."
    );

    debug_assert!(
        t_stride.is_power_of_two() && t_stride <= num_amps / 2,
        "Target stride must be a power of two and no larger than half the number of amplitudes."
    );

    debug_assert!(
        c_stride != t_stride,
        "Control and target strides must be different."
    );

    debug_assert!(
        matrix.size() == 2,
        "Controlled single-qubit gates must have dimension two."
    );

    let num_qubits = num_amps.ilog2() as usize;
    let control_qubit = num_qubits - 1 - c_stride.ilog2() as usize;
    let target_qubit = num_qubits - 1 - t_stride.ilog2() as usize;

    let identity = matrix::i();
    let p0 = matrix::p0();
    let p1 = matrix::p1();

    // Expand the controlled gate across the full system:
    //
    // U = M0 + M1
    //
    // M0 places P0 = |0⟩⟨0| on the control and identity elsewhere.
    //
    // M1 places P1 = |1⟩⟨1| on the control, `matrix` on the target,
    // and identity elsewhere.

    let m0_fn = |i: usize| {
        if control_qubit == i { &p0 }
        else { &identity }
    };

    let m1_fn = |i: usize| { 
        if control_qubit == i { &p1 }
        else if target_qubit == i { matrix }
        else { &identity }
    };

    let mut m0 = m0_fn(0).clone();
    let mut m1 = m1_fn(0).clone();

    for i in 1..num_qubits {
        m0 = kronecker_product(&m0, m0_fn(i));
        m1 = kronecker_product(&m1, m1_fn(i));
    }

    let mut output = vec![Complex64::ZERO; num_amps];

    for (row, result) in output.iter_mut().enumerate() {
        for (col, input) in amps.iter().enumerate() {
            let e0 = m0.get(row, col);
            let e1 = m1.get(row, col);
            *result += (e0 + e1) * input;
        }
    }

    amps.copy_from_slice(&output);
}

/// Returns the Kronecker product of two square matrices.
#[allow(dead_code)]
fn kronecker_product(a: &SquareMatrix, b: &SquareMatrix) -> SquareMatrix {
    let a_size = a.size();
    let b_size = b.size();
    
    let mut result = SquareMatrix::zero(a_size * b_size);

    for i in 0..a_size {
        for j in 0..a_size {
            for k in 0..b_size {
                for l in 0..b_size {
                    let row = i * b_size + k;
                    let col = j * b_size + l;

                    *result.get_mut(row, col) = a.get(i, j) * b.get(k, l)
                }
            }
        }
    }

    result
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
    fn strided_1q_unit_stride() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_1q_strided(&mut amps, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[2.0, 1.0, 4.0, 3.0]),
        );
    }

    #[test]
    fn strided_1q_larger_stride() {
        let mut amps =
            real_amps(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        apply_1q_strided(&mut amps, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]),
        );
    }

    #[test]
    fn strided_c2q_more_significant_control() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_c2q_strided(&mut amps, 2, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 2.0, 4.0, 3.0]),
        );
    }

    #[test]
    fn strided_c2q_more_significant_target() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_c2q_strided(&mut amps, 1, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 4.0, 3.0, 2.0]),
        );
    }

    #[test]
    fn strided_1q_complex_gate() {
        let mut amps = real_amps(&[1.0, 0.0]);

        apply_1q_strided(&mut amps, 1, &matrix::y());

        assert_amps_eq(
            &amps,
            &[
                Complex64::ZERO,
                Complex64::new(0.0, 1.0),
            ],
        );
    }

    #[test]
    fn strided_1q_preserves_norm() {
        let mut amps = real_amps(&[1.0, 0.0, 0.0, 0.0]);

        apply_1q_strided(&mut amps, 2, &matrix::h());

        let norm: f64 = amps.iter().map(Complex64::norm_sqr).sum();

        assert!((norm - 1.0).abs() < EPSILON);
    }

    #[test]
    #[allow(deprecated)]
    fn kronecker_1q_unit_stride() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_1q_kronecker(&mut amps, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[2.0, 1.0, 4.0, 3.0]),
        );
    }

    #[test]
    #[allow(deprecated)]
    fn kronecker_1q_larger_stride() {
        let mut amps =
            real_amps(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        apply_1q_kronecker(&mut amps, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]),
        );
    }

    #[test]
    #[allow(deprecated)]
    fn kronecker_c2q_more_significant_control() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_c2q_kronecker(&mut amps, 2, 1, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 2.0, 4.0, 3.0]),
        );
    }

    #[test]
    #[allow(deprecated)]
    fn kronecker_c2q_more_significant_target() {
        let mut amps = real_amps(&[1.0, 2.0, 3.0, 4.0]);

        apply_c2q_kronecker(&mut amps, 1, 2, &matrix::x());

        assert_amps_eq(
            &amps,
            &real_amps(&[1.0, 4.0, 3.0, 2.0]),
        );
    }
}