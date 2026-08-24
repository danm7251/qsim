use num_complex::Complex64;

use crate::linalg::SquareMatrix;

/// Applies `matrix` to every amplitude pair separated by `t_stride` using
/// fused multiply-add operations.
///
/// `t_stride` identifies the target qubit's stride within `amps`.
///
/// # Safety
///
/// The host CPU must support FMA.
///
/// # Panics
///
/// Panics if `t_stride` is zero or does not describe a valid partition of
/// `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "1 Qubit Gate Strided FMA"))]
pub fn apply_1q(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) -> () {
    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in offset..(offset + t_stride) {
            unsafe { apply_pair(amps, index_low, t_stride, matrix); };
        }
    }
}

/// Applies `matrix` to amplitude pairs where the control qubit is `|1⟩`,
/// using fused multiply-add operations.
///
/// `c_stride` and `t_stride` identify the control and target qubits'
/// respective strides within `amps`.
///
/// # Safety
///
/// The host CPU must support FMA.
///
/// # Panics
///
/// Panics if either stride is zero, the strides are equal, or they do not
/// describe valid qubits within `amps`.
#[cfg_attr(feature = "trace", tracing::instrument(skip(amps), name = "2 Qubit Gate Strided FMA"))]
pub fn apply_c2q(amps: &mut[Complex64], c_stride: usize, t_stride: usize, matrix: &SquareMatrix) -> () {
    if c_stride < t_stride {
        // Target is more significant, so select T=0 blocks before C=1 blocks.
        for t_block in (0..amps.len()).step_by(2 * t_stride) {
            let t_is_zero = t_block..(t_block + t_stride);

            for c_block in t_is_zero.step_by(2 * c_stride) {
                let c_is_one = (c_block + c_stride)..(c_block + 2 * c_stride);

                for index_low in c_is_one {
                    unsafe { apply_pair(amps, index_low, t_stride, matrix); };
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
                    unsafe { apply_pair(amps, index_low, t_stride, matrix); };
                }
            }
        }
    }
}

/// Applies `matrix` to the amplitude pair beginning at `index_low` using
/// fused multiply-add operations.
///
/// The paired amplitude is located at `index_low + t_stride`.
///
/// # Panics
///
/// Panics if either amplitude index is outside `amps`.
#[target_feature(enable = "fma")]
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
                coefficient.re.mul_add(
                    value.re,
                    (-coefficient.im).mul_add(value.im, total.re),
                ),
                coefficient.re.mul_add(
                    value.im,
                    coefficient.im.mul_add(value.re, total.im),
                ),
            );
        }

        output[i] = total;
    }

    amps[index_low] = output[0];
    amps[index_high] = output[1];
}