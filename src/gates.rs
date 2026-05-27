use std::f64::consts::SQRT_2;

use ndarray::{array, Array2};
use num_complex::Complex64;

use crate::math_utils::C64;

#[derive(Clone, Copy)]
pub enum Gate {
    I,
    X { target: usize },
    Y { target: usize },
    Z { target: usize },
    H { target: usize },
    S { target: usize },
    T { target: usize },
    CNOT { control: usize, target: usize }
}

impl Gate {
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Gate::I => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(1., 0.)]
            ],
            Gate::X {..} => array![
                [C64(0., 0.), C64(1., 0.)],
                [C64(1., 0.), C64(0., 0.)]
            ],
            Gate::Y {..} => array![
                [C64(0., 0.), C64(0., -1.)],
                [C64(0., 1.), C64(0., 0.)]
            ],
            Gate::Z {..} => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(-1., 0.)]
            ],
            Gate::H {..} => {
                let c = C64(1.0/SQRT_2, 0.);
                c*array![
                    [C64(1., 0.), C64(1., 0.)],
                    [C64(1., 0.), C64(-1., 0.)]
                ]
            },
            Gate::S {..} => array![
                [C64(1., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(0., 1.)]
            ],
            Gate::T {..} => {
                let c = 1.0/SQRT_2;
                array![
                    [C64(1., 0.), C64(0., 0.)],
                    [C64(0., 0.), C64(c, c)]
                ]
            },
            Gate::CNOT {..} => array![
                [C64(1., 0.), C64(0., 0.), C64(0., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(1., 0.), C64(0., 0.), C64(0., 0.)],
                [C64(0., 0.), C64(0., 0.), C64(0., 0.), C64(1., 0.)],
                [C64(0., 0.), C64(0., 0.), C64(1., 0.), C64(0., 0.)]
            ]
        }
    }
}