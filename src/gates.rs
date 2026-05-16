use std::f64::consts::SQRT_2;

use ndarray::{array, Array2};
use num_complex::Complex64;

use crate::math_utils::C64;

pub enum Gate64 {
    I,
    X,
    Y,
    Z,
    H,
    S,
    T
}

impl Gate64 {
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Gate64::I => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(1., 0.)]
            ],
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
            ],
            Gate64::H => {
                let c = C64(SQRT_2, 0.);
                c*array![
                    [C64(1., 0.), C64(1., 0.)],
                    [C64(1., 0.), C64(-1., 0.)]
                ]
            },
            Gate64::S => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(0., 1.)]
            ],
            Gate64::T => {
                let c = 1.0 / SQRT_2;
                array![
                    [C64(1., 0.), C64(0., 0.)],
                    [C64(0., 0.), C64(c, c)]
                ]
            }
        }
    }
}