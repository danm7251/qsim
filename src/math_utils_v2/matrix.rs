use super::Element;

#[derive(Debug)]
pub struct Matrix {
    elements: Vec<Element>,
    dim: usize
}

impl Matrix {
    pub fn get(&self, row: usize, col: usize) -> &Element {
        debug_assert!(row < self.dim && col < self.dim);
        &self.elements[row * self.dim + col]
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Element {
        debug_assert!(row < self.dim && col < self.dim);
        &mut self.elements[row * self.dim + col]
    }
}

pub fn i() -> Matrix {
    Matrix {
        elements: vec![
            Element::new(1., 0. ), Element::new(0., 0. ),
            Element::new(0., 0. ), Element::new(1., 0. )
        ],
        dim: 2
    }
}

pub fn x() -> Matrix {
    Matrix {
        elements: vec![
            Element::new(0., 0. ), Element::new(1., 0. ),
            Element::new(1., 0. ), Element::new(0., 0. )
        ],
        dim: 2
    }
}

pub fn y() -> Matrix {
    Matrix {
        elements: vec![
            Element::new(0., 0. ), Element::new(0., -1. ),
            Element::new(0., 1. ), Element::new(0., 0. )
        ],
        dim: 2
    }
}

pub fn z() -> Matrix {
    Matrix {
        elements: vec![
            Element::new(1., 0. ), Element::new(0., 0. ),
            Element::new(0., 0. ), Element::new(-1., 0. )
        ],
        dim: 2
    }
}