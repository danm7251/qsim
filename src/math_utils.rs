use std::fmt::Display;

use ndarray::{Array2, ScalarOperand, s};
use num_complex::{Complex32, Complex64, ComplexFloat};

// TODO: Decide on Complex::<F>::new() shorthand.
//       Keeping only C64 & C32 is an option but how to switch them?
//       Defining a c! macro that infers <F> is an option.

// TODO: Move print_matrix() into a formatting module.

// Shorthand for initialising Complex<F> types
pub const C64: fn(f64, f64) -> Complex64 = |re: f64, im: f64| Complex64::new(re, im);
pub const C32: fn(f32, f32) -> Complex32 = |re: f32, im: f32| Complex32::new(re, im);

// Returns the kronecker product of two matrices.
// Follows the naive yet faithful approach to the operation.
// Will be iterated on once benchmarking is implemented.
pub fn kron<F: ComplexFloat + ScalarOperand>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    // Gets the shapes of the input matrices.
    let (arows, acols) = a.dim();
    let (brows, bcols) = b.dim();

	// Applying the kronecker product to two matrices A(m x n) and B(p x q):
	// gives C(pm x qn). Here we define an empty matrix of shape C.
    let mut result = Array2::<F>::zeros((arows*brows, acols*bcols));

    for i in 0..arows {
        for j in 0..acols {
			// Loops through every element in A and multiplies it by the entire B matrix.
			// Note that ndarrays are indexed as [[i, j]] instead of [i][j].
            let block = b * a[[i, j]];

            // Retrieves a slice of the result and updates it.
            let mut target_slice = result.slice_mut(s![
                i*brows .. (i+1)*brows,
                j*bcols .. (j+1)*bcols
            ]);
            target_slice.assign(&block);
        }
    }
    result
}

// Helper function for formatting matrices.
pub fn print_matrix<F: ComplexFloat>(matrix: &Array2<F>) where F::Real: Display {
    // Defines the number of spaces between values.
    let value_gap = 3;

    // Generates the dynamic portion of the box header and footer.
    let (_, width) = matrix.dim();
    let bar = "─".repeat(width*11 + (width-1)*value_gap);

    // Output
    println!("┌─{}─┐", bar);
    for row in matrix.rows() {
        let formatted: Vec<String> = row.iter().map(|c| format!("{:+.2}{:+.2}i", c.re(), c.im())).collect();
        println!("│ {} │", formatted.join(&" ".repeat(value_gap)));
    }
    println!("└─{}─┘", bar);
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::math_utils::print_matrix;

    use super::{Array2, Complex64, kron, C64};

    #[test]
    fn test_kron() {
        struct TestCase {
            name: &'static str,
            input_a: Array2<Complex64>,
            input_b: Array2<Complex64>,
            expected: Array2<Complex64>
        }

        let cases: &[TestCase] = &[
            TestCase {
                name: "test_identity_2x2_squared",
                input_a: Array2::<Complex64>::eye(2),
                input_b: Array2::<Complex64>::eye(2),
                expected: Array2::<Complex64>::eye(4)
            },
            TestCase {
                name: "test_X_Z",
                input_a: array![
                    [C64(0., 0.), C64(1., 0.)],
                    [C64(1., 0.), C64(0., 0.)]
                ],
                input_b: array![
                    [C64(1., 0.), C64(0., 0.)],
                    [C64(0., 0.), C64(-1., 0.)]
                ],
                expected: array![
                    [C64(0., 0.), C64(0., 0.), C64(1., 0.), C64(0., 0.)],
                    [C64(0., 0.), C64(0., 0.), C64(0., 0.), C64(-1., 0.)],
                    [C64(1., 0.), C64(0., 0.), C64(0., 0.), C64(0., 0.)],
                    [C64(0., 0.), C64(-1., 0.), C64(0., 0.), C64(0., 0.)],
                ],
            },
            TestCase {
                name: "test_1x3_2x2",
                input_a: array![
                    [C64(1., 0.), C64(2., 0.), C64(3., 0.)]
                ],
                input_b: array![
                    [C64(0., 0.), C64(1., 0.)],
                    [C64(1., 0.), C64(0., 0.)]
                ],
                expected: array![
                    [C64(0., 0.), C64(1., 0.), C64(0., 0.), C64(2., 0.), C64(0., 0.), C64(3., 0.)],
                    [C64(1., 0.), C64(0., 0.), C64(2., 0.), C64(0., 0.), C64(3., 0.), C64(0., 0.)]
                ]
            },
            TestCase {
                name: "test_complex_numbers",
                input_a: array![
                    [C64(1., 2.), C64(0., -1.)],
                    [C64(3., 0.), C64(1., -2.)]
                ],
                input_b: array![
                    [C64(1., 0.), C64(2., -3.)],
                    [C64(0., 1.), C64(1., 1.) ]
                ],
                expected: array![
                    [C64(1., 2.),  C64(8., 1.),  C64(0., -1.),  C64(-3., -2.)],
                    [C64(-2., 1.), C64(-1., 3.),  C64(1., 0.),  C64(1., -1.) ],
                    [C64(3., 0.),  C64(6., -9.),  C64(1., -2.),  C64(-4., -7.) ],
                    [C64(0., 3.),  C64(3., 3.),   C64(2., 1.),  C64(3., -1.) ],
                ]
            }
        ];

        for case in cases {
            let result = kron(&case.input_a, &case.input_b);
            let (result_dim, expect_dim) = (result.dim(), case.expected.dim());
            if result_dim != expect_dim {
                println!("[\x1b[1;31mERROR\x1b[0m][{}]: Shapes do not match:\nExpected: {:?}\nResult: {:?}", case.name, expect_dim, result_dim);
                panic!("Mismatched shapes");
            }
            if result != case.expected {
                println!("[\x1b[1;31mERROR\x1b[0m][{}]: Matrices do not match", case.name);
                println!("Expected:");
                print_matrix(&case.expected);
                println!("Result:");
                print_matrix(&result);
                panic!()
            }
        }
    }
}