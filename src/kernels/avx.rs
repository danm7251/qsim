//! AVX state-vector gate kernels.

use std::simd::{Simd, simd_swizzle};

use num_complex::Complex64;

use crate::linalg::SquareMatrix;

type F64x4 = Simd<f64, 4>;

// KERNEL ENTRY POINTS

// Default entry point.
/// Applies a 2×2 matrix using the portable AVX kernel.
///
/// # Safety
///
/// The host CPU must support AVX. `t_stride` must describe a valid
/// state-vector partition and must be at least 2
#[target_feature(enable = "avx")]
pub unsafe fn apply_1q(amps: &mut [Complex64], t_stride: usize, matrix: &SquareMatrix) {
    apply_1q_portable(amps, t_stride, matrix);
}

// Supports kernel variant selection during benchmarking.
#[cfg(feature = "bench")]
#[derive(Clone, Copy, Debug)]
pub enum AvxVariant {
    Scalar,
    Portable,
}

// Benchmarking entry point. Can select based on `AvxVariant`.
#[cfg(feature = "bench")]
#[target_feature(enable = "avx")]
pub unsafe fn apply_1q_with_variant(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix, variant: AvxVariant) {
    match variant {
        AvxVariant::Scalar => {
            apply_1q_scalar(amps, t_stride, matrix);
        }
        AvxVariant::Portable => {
            apply_1q_portable(amps, t_stride, matrix);
        }
    }
}

// KERNELS

// Scalar Kernel.
// Utilises the generic scalar kernel but with AVX features enabled.

#[cfg(feature = "bench")]
#[target_feature(enable = "avx")]
fn apply_1q_scalar(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) {
    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in offset..offset + t_stride {
            apply_pair_scalar(amps, index_low, t_stride, matrix);
        }
    }
}

#[cfg(feature = "bench")]
#[target_feature(enable = "avx")]
fn apply_pair_scalar(amps: &mut[Complex64], index_low: usize, t_stride: usize, matrix: &SquareMatrix) {
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

// Portable Kernel
// Utilises the nightly `portable_simd` feature to use SIMD explicitly.

/// A representation of 8 256-bit SIMD vectors, storing the real and imaginary coefficients of a matrix.
struct SplattedMatrix {
    m00_re: F64x4,
    m00_im: F64x4,
    m01_re: F64x4,
    m01_im: F64x4,
    m10_re: F64x4,
    m10_im: F64x4,
    m11_re: F64x4,
    m11_im: F64x4,
}

impl SplattedMatrix {
    #[inline(always)]
    fn new(matrix: &SquareMatrix) -> Self {
        let m00 = matrix.get(0, 0);
        let m01 = matrix.get(0, 1);
        let m10 = matrix.get(1, 0);
        let m11 = matrix.get(1, 1);

        // Splat each f64 component across separate vectors.
        Self {
            m00_re: F64x4::splat(m00.re),
            m00_im: F64x4::splat(m00.im),
            m01_re: F64x4::splat(m01.re),
            m01_im: F64x4::splat(m01.im),
            m10_re: F64x4::splat(m10.re),
            m10_im: F64x4::splat(m10.im),
            m11_re: F64x4::splat(m11.re),
            m11_im: F64x4::splat(m11.im),
        }
    }
}

#[target_feature(enable = "avx")]
fn apply_1q_portable(amps: &mut[Complex64], t_stride: usize, matrix: &SquareMatrix) {
    debug_assert!(t_stride >= 2);
    debug_assert_eq!(t_stride % 2, 0);

    // Perpare invariant values in vectors.
    let matrix = SplattedMatrix::new(matrix);
    let signs = F64x4::from_array([-1.0, 1.0, -1.0, 1.0]);

    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in (offset..offset + t_stride).step_by(2) {
            apply_pair_portable(amps, index_low, t_stride, &matrix, signs);
        }
    }
}

#[target_feature(enable = "avx")]
fn apply_pair_portable(amps: &mut[Complex64], index_low: usize, t_stride: usize, matrix: &SplattedMatrix, signs: F64x4) {
    let index_high = index_low + t_stride;

    let low_inputs = Simd::from_array([
        amps[index_low].re,
        amps[index_low].im,
        amps[index_low + 1].re,
        amps[index_low + 1].im,
    ]);

    let high_inputs = Simd::from_array([
        amps[index_high].re,
        amps[index_high].im,
        amps[index_high + 1].re,
        amps[index_high + 1].im,
    ]);

    let temp1 = mul_complex_portable(low_inputs, matrix.m00_re, matrix.m00_im, signs);
    let temp2 = mul_complex_portable(high_inputs, matrix.m01_re, matrix.m01_im, signs);
    let low_outputs = temp1 + temp2;

    let temp1 = mul_complex_portable(low_inputs, matrix.m10_re, matrix.m10_im, signs);
    let temp2 = mul_complex_portable(high_inputs, matrix.m11_re, matrix.m11_im, signs);
    let high_outputs = temp1 + temp2;
    
    let low_outputs = low_outputs.to_array();
    let high_outputs = high_outputs.to_array();

    amps[index_low] = Complex64::new(low_outputs[0], low_outputs[1]);
    amps[index_low + 1] = Complex64::new(low_outputs[2], low_outputs[3]);
    amps[index_high] = Complex64::new(high_outputs[0], high_outputs[1]);
    amps[index_high + 1] = Complex64::new(high_outputs[2], high_outputs[3])
}

#[target_feature(enable = "avx")]
fn mul_complex_portable(input: F64x4, re: F64x4, im: F64x4, signs: F64x4) -> F64x4 {
    let re_products = input * re;
    let im_products = input * im;

    // Line up imaginary lanes with real lanes.
    let sw_im_products = simd_swizzle!(im_products, [1, 0, 3, 2]);

    re_products + sw_im_products * signs
}