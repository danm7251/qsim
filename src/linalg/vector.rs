use super::Element;

#[derive(Debug)]
pub struct Vector {
    elements: Vec<Element>
}

impl Vector {
    pub fn zeros(len: usize) -> Self {
        Self {
            elements: vec![Element::ZERO; len]
        }
    }

    pub fn get(&self, index: usize) -> &Element {
        debug_assert!(index < self.elements.len());
        &self.elements[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut Element {
        debug_assert!(index < self.elements.len());
        &mut self.elements[index]
    }

    #[allow(unused)]
    pub fn as_slice(&self) -> &[Element] {
        &self.elements
    }

    #[allow(unused)]
    pub fn as_mut_slice(&mut self) -> &mut [Element] {
        &mut self.elements
    }
}