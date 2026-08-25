pub(crate) mod avx;
pub(crate) mod fma;
pub(crate) mod generic;

// Allows benchmarking experimental AVX kernels.
#[cfg(feature = "bench")]
pub use avx::{
    apply_1q_with_variant as apply_1q_avx_with_variant,
    AvxVariant,
};

// Allows benchmarking the generic kernel directly.
#[cfg(feature = "bench")]
pub use generic::apply_1q as apply_1q_generic;

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::linalg::SquareMatrix;

    use super::{avx, generic};

    #[test]
    fn portable_avx_matches_generic() {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }

        let matrix = SquareMatrix::from_array([
            [(0.25, -0.50), (0.75, 0.125)],
            [(-0.375, 0.625), (0.50, -0.25)],
        ]);

        let input = [
            Complex64::new(0.10, -0.20),
            Complex64::new(0.30, 0.40),
            Complex64::new(-0.50, 0.60),
            Complex64::new(0.70, -0.80),
            Complex64::new(-0.15, 0.25),
            Complex64::new(0.35, -0.45),
            Complex64::new(0.55, 0.65),
            Complex64::new(-0.75, 0.85),
        ];

        let mut expected = input;
        let mut actual = input;

        generic::apply_1q(&mut expected, 2, &matrix);

        unsafe {
            avx::apply_1q(&mut actual, 2, &matrix);
        }

        for (index, (actual, expected)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            let difference = (*actual - *expected).norm();

            assert!(
                difference < 1e-12,
                "amplitude {index} differs: \
                AVX={actual}, generic={expected}, \
                difference={difference}",
            );
        }
    }
}