pub(crate) mod avx;
pub(crate) mod avx_fma;
pub(crate) mod fma;
pub(crate) mod portable;

// Allows benchmarking the generic kernel directly.
#[cfg(feature = "bench")]
pub use avx::apply_1q as apply_1q_avx;
pub use fma::apply_1q as apply_1q_fma;
#[allow(deprecated)]
pub use portable::{apply_1q_kronecker, apply_1q_strided, apply_c2q_kronecker, apply_c2q_strided};

#[cfg(test)]
mod test_utils {
    use num_complex::Complex64;

    use crate::linalg::SquareMatrix;

    // Floating-point error tolerance.
    pub(crate) const EPSILON: f64 = 1e-14;

    // Hardware capability assertions.

    pub(crate) fn require_avx() {
        assert!(
            is_x86_feature_detected!("avx"),
            "Host CPU does not support AVX!"
        );
    }

    pub(crate) fn require_fma() {
        assert!(
            is_x86_feature_detected!("fma"),
            "Host CPU does not support FMA!"
        );
    }

    /// Deterministic pseudo-random amplitudes from -1..1, generated with an LCG.
    /// The same `(len, seed)` always gives the same output.
    pub(crate) fn random_amps(len: usize, seed: u64) -> Vec<Complex64> {
        // 64-bit LCG with modulus 2^64. Constants are from the "Parameters in common use"
        // table in Wikipedia's "Linear congruential generator" article,
        // in the row "rsixfour.c, from the MMIX Supplement".
        const LCG_MULTIPLIER: u64 = 6364136223846793005;
        const LCG_INCREMENT: u64 = 9754186451795953191;

        let mut lcg = seed;
        let mut next = || {
            // Advance generator.
            lcg = lcg.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);

            // Scale to 0..=1.
            let unit = lcg as f64 / u64::MAX as f64;

            // Rescale to -1..=1.
            2.0 * unit - 1.0
        };

        (0..len)
            .map(|_| Complex64::new(next(), next()))
            .collect()
    }

    pub(crate) fn random_matrix(seed: u64) -> SquareMatrix {
        let c = random_amps(4, seed);
        SquareMatrix::from_array([
            [(c[0].re, c[0].im), (c[1].re, c[1].im)],
            [(c[2].re, c[2].im), (c[3].re, c[3].im)],
        ])
    }

    pub(crate) fn assert_amps_eq(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());

        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let difference = (*actual - *expected).norm();

            assert!(
                difference < EPSILON,
                "Amplitude {index} differs: {actual} != {expected}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        kernels::test_utils::{
            assert_amps_eq,
            random_amps, 
            random_matrix, 
            require_avx, 
            require_fma
        }
    };

    use super::{avx, avx_fma, portable};

    #[test]
    fn avx_1q_matches_portable() {
        require_avx();

        let n = 7;
        let t_qubit = 0..(n - 1); // AVX does not apply to T = N - 1.
        for t in t_qubit {
            let t_stride = 1 << (n - t - 1);
            let input = random_amps(1 << n, t);
            let matrix = random_matrix(t + 100);

            let mut expected = input.clone();
            portable::apply_1q_strided(&mut expected, t_stride, &matrix);

            let mut output = input;
            unsafe {
                avx::apply_1q(&mut output, t_stride, &matrix);
            }

            assert_amps_eq(&output, &expected);
        }
    }

    #[test]
    fn avx_c2q_matches_portable() {
        require_avx();

        let n = 7;
        for c in 0..(n - 1) {
            for t in 0..(n - 1) {
                if c == t {
                    continue;
                }

                let t_stride = 1 << (n - t - 1);
                let c_stride = 1 << (n - c - 1);

                let input = random_amps(1 << n, c + t);
                let matrix = random_matrix(c + t + 100);

                let mut expected = input.clone();
                portable::apply_c2q_strided(&mut expected, c_stride, t_stride, &matrix);

                let mut output = input;
                unsafe {
                    avx::apply_c2q(&mut output, c_stride, t_stride, &matrix);
                }

                assert_amps_eq(&output, &expected);
            }
        }
    }

    #[test]
    fn avx_fma_1q_matches_portable() {
        require_avx();
        require_fma();

        let n = 7;
        let t_qubit = 0..(n - 1); // AVX does not apply to T = N - 1.
        for t in t_qubit {
            let t_stride = 1 << (n - t - 1);
            let input = random_amps(1 << n, t);
            let matrix = random_matrix(t + 100);

            let mut expected = input.clone();
            portable::apply_1q_strided(&mut expected, t_stride, &matrix);

            let mut output = input;
            unsafe {
                avx_fma::apply_1q(&mut output, t_stride, &matrix);
            }

            assert_amps_eq(&output, &expected);
        }
    }

    #[test]
    fn avx_fma_c2q_matches_portable() {
        require_avx();
        require_fma();

        let n = 7;
        for c in 0..(n - 1) {
            for t in 0..(n - 1) {
                if c == t {
                    continue;
                }

                let t_stride = 1 << (n - t - 1);
                let c_stride = 1 << (n - c - 1);

                let input = random_amps(1 << n, c + t);
                let matrix = random_matrix(c + t + 100);

                let mut expected = input.clone();
                portable::apply_c2q_strided(&mut expected, c_stride, t_stride, &matrix);

                let mut output = input;
                unsafe {
                    avx_fma::apply_c2q(&mut output, c_stride, t_stride, &matrix);
                }

                assert_amps_eq(&output, &expected);
            }
        }
    }
}
