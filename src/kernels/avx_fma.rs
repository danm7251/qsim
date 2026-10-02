//! AVX + FMA state-vector gate kernels.

use std::simd::{Simd, StdFloat, simd_swizzle};

use num_complex::Complex64;

use crate::linalg::SquareMatrix;

type F64x4 = Simd<f64, 4>;

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

#[target_feature(enable = "avx,fma")]
pub unsafe fn apply_1q(amps: &mut [Complex64], t_stride: usize, matrix: &SquareMatrix) {
    debug_assert!(t_stride >= 2);
    debug_assert_eq!(t_stride % 2, 0);

    // Perpare invariant values in vectors.
    let matrix = SplattedMatrix::new(matrix);
    let signs = F64x4::from_array([-1.0, 1.0, -1.0, 1.0]);

    for offset in (0..amps.len()).step_by(2 * t_stride) {
        for index_low in (offset..offset + t_stride).step_by(2) {
            apply_pair(amps, index_low, t_stride, &matrix, signs);
        }
    }
}

#[target_feature(enable = "avx,fma")]
fn apply_pair(
    amps: &mut [Complex64],
    index_low: usize,
    t_stride: usize,
    matrix: &SplattedMatrix,
    signs: F64x4,
) {
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

#[target_feature(enable = "avx,fma")]
fn mul_complex_portable(input: F64x4, re: F64x4, im: F64x4, signs: F64x4) -> F64x4 {
    let re_products = input * re;
    let im_products = input * im;

    // Line up imaginary lanes with real lanes.
    let sw_im_products = simd_swizzle!(im_products, [1, 0, 3, 2]);

    signs.mul_add(sw_im_products, re_products)
}
