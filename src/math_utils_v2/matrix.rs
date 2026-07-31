use super::Element;

#[derive(Debug)]
pub struct SquareMatrix {
    elements: Vec<Element>,
    dim: usize
}

impl SquareMatrix {
    pub fn zero(size: usize) -> Self {
        Self {
            elements: vec![Element::ZERO; size * size],
            dim: size
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get(&self, row: usize, col: usize) -> &Element {
        debug_assert!(row < self.dim && col < self.dim);
        &self.elements[row * self.dim + col]
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Element {
        debug_assert!(row < self.dim && col < self.dim);
        &mut self.elements[row * self.dim + col]
    }
}

pub fn i() -> SquareMatrix {
    SquareMatrix {
        elements: vec![
            Element::new(1., 0. ), Element::new(0., 0. ),
            Element::new(0., 0. ), Element::new(1., 0. )
        ],
        dim: 2
    }
}

pub fn x() -> SquareMatrix {
    SquareMatrix {
        elements: vec![
            Element::new(0., 0. ), Element::new(1., 0. ),
            Element::new(1., 0. ), Element::new(0., 0. )
        ],
        dim: 2
    }
}

pub fn y() -> SquareMatrix {
    SquareMatrix {
        elements: vec![
            Element::new(0., 0. ), Element::new(0., -1. ),
            Element::new(0., 1. ), Element::new(0., 0. )
        ],
        dim: 2
    }
}

pub fn z() -> SquareMatrix {
    SquareMatrix {
        elements: vec![
            Element::new(1., 0. ), Element::new(0., 0. ),
            Element::new(0., 0. ), Element::new(-1., 0. )
        ],
        dim: 2
    }
}