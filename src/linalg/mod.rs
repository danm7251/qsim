pub mod matrix;
pub mod vector;

pub use matrix::SquareMatrix;
pub use vector::Vector;

use num_complex::Complex;

// Floating point precision parameter.
pub type Element = Complex<f64>;

// Multi-type operations.

/// Multiplies a vector by a matrix allocating a new vector of equal length.
pub fn linear_map(matrix: &SquareMatrix, vector: &Vector) -> Vector {
    debug_assert!(vector.len() == matrix.dim());

    let n = vector.len();
    let mut output = Vector::zeros(n);

    for i in 0..n {
        let mut total = Element::ZERO;

        for j in 0..n {
            total += matrix.get(i, j) * vector.get(j);
        }

        *output.get_mut(i) = total;
    }

    output
}