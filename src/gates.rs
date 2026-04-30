use ndarray::{array, Array2};
use num_complex::Complex64;

use crate::math_utils::C64;

enum Gate64 {
    X,
    Y,
    Z
}

impl Gate64 {
    fn matrix(&self) -> Array2<Complex64> {
        match self {
            Gate64::X => array![
                [C64(0., 0.), C64(1., 0.)],
                [C64(1., 0.), C64(0., 0.)]
            ],
            Gate64::Y => array![
                [C64(0., 0.), C64(0., -1.)],
                [C64(0., 1.), C64(0., 0.)]
            ],
            Gate64::Z => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(-1., 0.)]
            ]
        }
    }
}