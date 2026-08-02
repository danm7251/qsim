pub mod matrix;
pub mod vector;

pub use matrix::SquareMatrix;
pub use vector::Vector;

use num_complex::Complex;

// Floating point precision parameter.
pub type Element = Complex<f64>;