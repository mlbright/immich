use crate::schemas::Shape;

#[derive(Clone, Debug, PartialEq)]
pub struct ShapePolicy {
    pub dims: Vec<Shape>,
    pub label: String,
}

impl Default for ShapePolicy {
    fn default() -> Self {
        Self {
            dims: vec![Shape::new(1)],
            label: String::new(),
        }
    }
}

impl ShapePolicy {
    pub fn new(dims: Vec<Shape>) -> Self {
        Self {
            dims,
            label: String::new(),
        }
    }

    pub fn with_label(dims: Vec<Shape>, label: String) -> Self {
        Self { dims, label }
    }

    pub fn pinned(&self) -> Shape {
        let first = self.dims[0];
        Shape {
            batch: 1,
            height: if self.dims.iter().all(|shape| shape.height == first.height) {
                first.height
            } else {
                None
            },
            width: if self.dims.iter().all(|shape| shape.width == first.width) {
                first.width
            } else {
                None
            },
        }
    }
}

pub fn batches(maximum: usize) -> Vec<i64> {
    if maximum > 1 {
        vec![1, maximum as i64]
    } else {
        vec![1]
    }
}

pub fn runs(total: usize, batches: &[i64]) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut remaining = total as i64;
    while remaining > 0 {
        let size = batches
            .iter()
            .copied()
            .filter(|size| *size <= remaining)
            .max()
            .unwrap_or(1);
        runs.push(size as usize);
        remaining -= size;
    }
    runs
}
