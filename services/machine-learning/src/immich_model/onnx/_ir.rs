use anyhow::Result;

use crate::onnx_ir::{Model, lift_constants_to_initializers};

pub fn canonicalize_constants(model: &mut Model) -> Result<bool> {
    lift_constants_to_initializers(model, 0)
}
