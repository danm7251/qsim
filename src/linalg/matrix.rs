use super::Element;

#[derive(Debug)]
pub struct SquareMatrix {
    elements: Vec<Element>,
    size: usize
}

impl SquareMatrix {
    pub fn zero(size: usize) -> Self {
        Self {
            elements: vec![Element::ZERO; size * size],
            size
        }
    }

    /// Constructs an NxN `SquareMatrix` from a 2D array of tuples.
    pub fn from_array<const N: usize>(array: [[(f64, f64); N]; N]) -> Self {
        let mut elements: Vec<Element> = Vec::with_capacity(N *N);
        for row in array {
            for (re, im) in row {
                elements.push(Element::new(re, im));
            }
        }
        
        Self {
            elements,
            size: N
        }
    }

    #[warn(deprecated)]
    #[deprecated]
    /// Significantly slower than `from_array`.
    pub fn from_array_2<const N: usize>(elements: [[Element; N]; N]) -> Self {
        Self {
            elements: elements.into_iter().flatten().collect(),
            size: N
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get(&self, row: usize, col: usize) -> &Element {
        debug_assert!(row < self.size && col < self.size);
        &self.elements[row * self.size + col]
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Element {
        debug_assert!(row < self.size && col < self.size);
        &mut self.elements[row * self.size + col]
    }
}

pub fn i() -> SquareMatrix {
    SquareMatrix::from_array([
        [(1., 0. ), (0., 0. )],
        [(0., 0. ), (1., 0. )]
    ])
}

pub fn x() -> SquareMatrix {
    SquareMatrix::from_array([
            [(0., 0. ), (1., 0. )],
            [(1., 0. ), (0., 0. )]
    ])
}

pub fn y() -> SquareMatrix {
    SquareMatrix::from_array([
            [(0., 0. ), (0., -1. )],
            [(0., 1. ), (0., 0. )]
    ])
}

pub fn z() -> SquareMatrix {
    SquareMatrix::from_array([
            [(1., 0. ), (0., 0. )],
            [(0., 0. ), (-1., 0. )]
    ])
}