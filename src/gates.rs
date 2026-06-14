use std::f64::consts::SQRT_2;

use ndarray::{array, Array2};
use num_complex::Complex64;

use crate::math_utils::C64;

// At some point I intend to provide a more user facing enum,
// such as Instruction that encodes the target/control data.
// Then internally this can delegate to a more structured enum tree,
// E.g Gate::ControlledOp::OneControl::CNOT with target/control data,
// which I can use to route operations internally.
// However here Rust's strictness holds me back it will require some research,
// since I hope to avoid writing one giant match table for every single variant.

#[derive(Clone, Copy, Debug)]
pub enum Gate {
    I,
    X { target: usize },
    Y { target: usize },
    Z { target: usize },
    H { target: usize },
    S { target: usize },
    T { target: usize },
    CNOT { control: usize, target: usize },
    CRP { control: usize, target: usize, phi: f64 }
}

// Currently one qubit control, two qubit gates return their sub-operation.
// For example CX.matrix() == X.matrix().
// This is because encoding the control behaviour via the matrix is largely inneficient.
// The kernel handles the control behaviour internally.
// A representation that purely encodes the matrices is desirable here.

impl Gate {
    pub fn matrix(&self) -> Array2<Complex64> {
        // The reason the matrices are attached in this form is that,
        // ndarray makes it fairly difficult to use their arrays statically.
        // So I have to use constructors that act on static floats instead.
        // This added to the memory contiguity errors that ndarray cannot guarantee against,
        // is good motivation for writing my own implementation.
        // Additionally I do not need the N-dimensionality that ndarray provides,
        // only column vectors and matrices.
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
                [C64(0., 0.), C64(1., 0.)],
                [C64(1., 0.), C64(0., 0.)]
            ],
            Gate::CRP { phi , .. } => {
                let c = Complex64::from_polar(1., *phi);
                array![
                    [C64(1., 0.), C64(0., 0.)],
                    [C64(0., 0.), c]
                ]
            }
        }
    }
}