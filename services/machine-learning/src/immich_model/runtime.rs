use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};

use super::onnx::{_ir, lowering};
use crate::onnx_ir::{self, Model};

pub const RKNPU: &str = "RKNPU";
pub const REWRITE_SET_VERSION: u32 = 22;
pub const PACKAGE_VERSION: &str = "0.2.0";

const COREML: &str = "CoreMLExecutionProvider";
const CPU: &str = "CPUExecutionProvider";
const CUDA: &str = "CUDAExecutionProvider";
const MIGRAPHX: &str = "MIGraphXExecutionProvider";
const OPENVINO: &str = "OpenVINOExecutionProvider";
const TENSORRT: &str = "TensorrtExecutionProvider";
const TENSORRT_RTX: &str = "NvTensorRTRTXExecutionProvider";

type Transform = fn(&mut Model) -> Result<bool>;

pub struct RewriteContext {
    pub target: String,
    pub ort_version: Vec<u32>,
}

pub struct Rewrite {
    pub name: &'static str,
    pub gates: &'static [&'static str],
    transform: Option<Transform>,
}

impl Rewrite {
    const fn new(name: &'static str, gates: &'static [&'static str]) -> Self {
        Self {
            name,
            gates,
            transform: None,
        }
    }

    pub fn applies(&self, context: &RewriteContext) -> bool {
        self.gates.contains(&context.target.as_str())
    }
}

pub struct RewritePlan {
    pub rewrites: Vec<&'static Rewrite>,
    pub digest: String,
}

impl RewritePlan {
    pub fn names(&self) -> Vec<&'static str> {
        self.rewrites.iter().map(|rewrite| rewrite.name).collect()
    }
}

pub static REGISTRY: [Rewrite; 31] = [
    Rewrite::new("floatify_pad_mask", &[RKNPU]),
    Rewrite::new("pad_attention_heads", &[OPENVINO]),
    Rewrite::new(
        "decompose_attention",
        &[COREML, MIGRAPHX, OPENVINO, TENSORRT, TENSORRT_RTX, RKNPU],
    ),
    Rewrite::new("broadcast_shape_workaround", &[MIGRAPHX]),
    Rewrite::new("decompose_reduce_l2", &[COREML]),
    Rewrite::new("decompose_prelu", &[MIGRAPHX]),
    Rewrite::new("symmetrize_conv_pads", &[MIGRAPHX]),
    Rewrite::new("unpack_scrfd_heads", &[COREML, RKNPU]),
    Rewrite::new("greedy_ctc_topk", &[CUDA, TENSORRT, TENSORRT_RTX]),
    Rewrite::new("hard_swish", &[CUDA, OPENVINO]),
    Rewrite::new("im2col_patchify", &[CUDA, TENSORRT, TENSORRT_RTX, OPENVINO]),
    Rewrite::new("im2col_patchify_ragged", &[OPENVINO]),
    Rewrite::new("im2col_patchify_batch1", &[RKNPU]),
    Rewrite {
        name: "skip_layer_norm",
        gates: &[CPU, CUDA],
        transform: Some(lowering::fuse_skip_layer_norm),
    },
    Rewrite::new("nchw_image_input", &[COREML]),
    Rewrite::new("uint8_image_input", &[RKNPU]),
    Rewrite::new("float_image_input", &[RKNPU]),
    Rewrite::new("host_ctc_decode", &[OPENVINO]),
    Rewrite::new("raw_ctc_logits", &[RKNPU]),
    Rewrite::new("floatify_not_equal", &[RKNPU]),
    Rewrite::new("floatify_pad_keep", &[RKNPU]),
    Rewrite::new("host_token_embedding", &[RKNPU]),
    Rewrite::new("opaque_zero_mul", &[RKNPU]),
    Rewrite::new("fold_gather_elements", &[RKNPU]),
    Rewrite::new("decompose_gelu", &[RKNPU]),
    Rewrite::new("split_large_reduction", &[RKNPU]),
    Rewrite::new("fold_concat_into_conv", &[RKNPU]),
    Rewrite::new("split_large_conv_reduction", &[RKNPU]),
    Rewrite::new("split_deep_reduction", &[COREML]),
    Rewrite::new("pin_opset", &[RKNPU]),
    Rewrite::new("reinfer_shapes", &[RKNPU]),
];

fn py_tuple(values: &[u32]) -> String {
    match values {
        [single] => format!("({single},)"),
        _ => format!(
            "({})",
            values
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_list(names: &[&str]) -> String {
    format!(
        "[{}]",
        names
            .iter()
            .map(|name| format!("'{name}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn plan_rewrites(context: &RewriteContext) -> RewritePlan {
    let rewrites: Vec<&'static Rewrite> = REGISTRY
        .iter()
        .filter(|rewrite| rewrite.applies(context))
        .collect();
    let names: Vec<&str> = rewrites.iter().map(|rewrite| rewrite.name).collect();
    let facts = format!(
        "{PACKAGE_VERSION}\x1f{REWRITE_SET_VERSION}\x1f{}\x1f{}\x1f{}",
        context.target,
        py_tuple(&context.ort_version),
        py_list(&names)
    );
    let digest = Sha256::digest(facts.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()[..12]
        .to_owned();
    RewritePlan { rewrites, digest }
}

pub fn apply_rewrites(src_path: &Path, plan: &RewritePlan) -> Result<PathBuf> {
    let stem = src_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let out_path = src_path.with_file_name(format!("{stem}.rw-{}.onnx", plan.digest));
    if plan.rewrites.is_empty() {
        return Ok(src_path.to_path_buf());
    }
    let mut model = onnx_ir::load(src_path)?;
    let mut modified = false;
    for rewrite in &plan.rewrites {
        let Some(transform) = rewrite.transform else {
            bail!("The {} rewrite is not ported yet", rewrite.name);
        };
        modified |= transform(&mut model)?;
    }
    if !modified {
        return Ok(src_path.to_path_buf());
    }
    _ir::canonicalize_constants(&mut model)?;
    onnx_ir::remove_unused_nodes(&mut model);
    onnx_ir::deduplicate_initializers(&mut model, 1024)?;
    onnx_ir::topological_sort(&mut model)?;
    let tmp = out_path.with_extension("tmp");
    onnx_ir::save(&model, &tmp)?;
    fs::rename(&tmp, &out_path)?;
    Ok(out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_plan_matches_python_digest() {
        let plan = plan_rewrites(&RewriteContext {
            target: CPU.to_owned(),
            ort_version: vec![1, 31, 0],
        });
        assert_eq!(plan.names(), ["skip_layer_norm"]);
        assert_eq!(plan.digest, "ea7bc116b3ab");
    }
}
