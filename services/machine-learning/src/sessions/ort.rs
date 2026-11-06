use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::ptr::{self, NonNull};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use ndarray::{ArrayD, IxDyn};
use ort::AsPointer;
use ort::environment::Environment;
use ort::ep::CPU;
use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel, SessionBuilder};
use ort::value::{DynValue, TensorElementType, ValueType};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::{debug, info};

use super::policy::ShapePolicy;
use crate::config::settings;
use crate::immich_model::runtime::{RewriteContext, RewritePlan, plan_rewrites};
use crate::models::constants::SUPPORTED_PROVIDERS;
use crate::schemas::Shape;

pub const DYNAMIC_PROVIDERS: [&str; 1] = ["CPUExecutionProvider"];
pub const UNREADABLE: i32 = 3;
const PLANNED_AS: [(&str, &str); 1] = [("nv_tensorrt_rtx", "NvTensorRTRTXExecutionProvider")];

static ENVIRONMENT: LazyLock<Environment> = LazyLock::new(|| {
    ort::init()
        .with_name("immich_ml")
        .build()
        .expect("ONNX Runtime environment")
});

#[derive(Clone, Debug)]
pub enum Tensor {
    F32(ArrayD<f32>),
    F16(ArrayD<f16>),
    U8(ArrayD<u8>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(array) => array.shape(),
            Self::F16(array) => array.shape(),
            Self::U8(array) => array.shape(),
            Self::I32(array) => array.shape(),
            Self::I64(array) => array.shape(),
        }
    }

    fn into_value(self) -> Result<DynValue> {
        Ok(match self {
            Self::F32(array) => ort::value::Tensor::from_array(array)?.into_dyn(),
            Self::F16(array) => ort::value::Tensor::from_array(array)?.into_dyn(),
            Self::U8(array) => ort::value::Tensor::from_array(array)?.into_dyn(),
            Self::I32(array) => ort::value::Tensor::from_array(array)?.into_dyn(),
            Self::I64(array) => ort::value::Tensor::from_array(array)?.into_dyn(),
        })
    }

    fn zeros(ty: &str, shape: Vec<usize>) -> Result<Self> {
        let shape = IxDyn(&shape);
        Ok(match ty {
            "tensor(float)" => Self::F32(ArrayD::zeros(shape)),
            "tensor(float16)" => Self::F16(ArrayD::from_elem(shape, f16::ZERO)),
            "tensor(uint8)" => Self::U8(ArrayD::zeros(shape)),
            "tensor(int32)" => Self::I32(ArrayD::zeros(shape)),
            "tensor(int64)" => Self::I64(ArrayD::zeros(shape)),
            other => bail!("unsupported input type {other}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Size(i64),
    Symbol(String),
}

#[derive(Clone, Debug)]
pub struct SessionNode {
    pub name: String,
    pub ty: String,
    pub shape: Vec<Dim>,
}

impl SessionNode {
    fn from_outlet(name: &str, dtype: &ValueType) -> Self {
        match dtype {
            ValueType::Tensor {
                ty,
                shape,
                dimension_symbols,
            } => Self {
                name: name.to_owned(),
                ty: type_name(*ty).to_owned(),
                shape: shape
                    .iter()
                    .zip(dimension_symbols.iter())
                    .map(|(size, symbol)| {
                        if *size >= 0 {
                            Dim::Size(*size)
                        } else {
                            Dim::Symbol(symbol.to_string())
                        }
                    })
                    .collect(),
            },
            other => Self {
                name: name.to_owned(),
                ty: format!("{other}"),
                shape: Vec::new(),
            },
        }
    }
}

fn type_name(ty: TensorElementType) -> &'static str {
    match ty {
        TensorElementType::Float32 => "tensor(float)",
        TensorElementType::Float16 => "tensor(float16)",
        TensorElementType::Uint8 => "tensor(uint8)",
        TensorElementType::Int32 => "tensor(int32)",
        TensorElementType::Int64 => "tensor(int64)",
        _ => "tensor(unsupported)",
    }
}

fn label(pins: &[(String, i64)]) -> String {
    pins.iter()
        .map(|(name, size)| format!("{name}{size}"))
        .collect::<Vec<_>>()
        .join("_")
}

pub struct OrtGraph {
    session: Session,
    inputs: Vec<SessionNode>,
    outputs: Vec<SessionNode>,
    metadata: HashMap<String, String>,
    pub normalizes_input: bool,
    pub channels_first: bool,
}

impl OrtGraph {
    fn new(spec: &GraphSpec) -> Result<Self> {
        let graph = prepared(spec)?;
        let session = spec.session(&graph, None)?;
        let inputs: Vec<SessionNode> = session
            .inputs()
            .iter()
            .map(|input| SessionNode::from_outlet(input.name(), input.dtype()))
            .collect();
        let outputs: Vec<SessionNode> = session
            .outputs()
            .iter()
            .map(|output| SessionNode::from_outlet(output.name(), output.dtype()))
            .collect();
        let metadata = {
            let metadata = session.metadata()?;
            metadata
                .custom_keys()?
                .into_iter()
                .filter_map(|key| metadata.custom(&key).map(|value| (key, value)))
                .collect()
        };
        let image = inputs.first().context("graph has no inputs")?;
        let normalizes_input = image.ty == "tensor(uint8)";
        let channels_first =
            normalizes_input && image.shape.len() == 4 && image.shape[1] == Dim::Size(3);
        Ok(Self {
            session,
            inputs,
            outputs,
            metadata,
            normalizes_input,
            channels_first,
        })
    }

    pub fn get_inputs(&self) -> &[SessionNode] {
        &self.inputs
    }

    pub fn get_outputs(&self) -> &[SessionNode] {
        &self.outputs
    }

    pub fn get_metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn run(&self, input_feed: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        let input_feed = if self.channels_first {
            input_feed
                .into_iter()
                .map(|(name, frames)| match frames {
                    Tensor::U8(frames) => (
                        name,
                        Tensor::U8(
                            frames
                                .permuted_axes(IxDyn(&[0, 3, 1, 2]))
                                .as_standard_layout()
                                .into_owned(),
                        ),
                    ),
                    other => (name, other),
                })
                .collect()
        } else {
            input_feed
        };
        self.infer(input_feed)
    }

    pub fn infer(&self, input_feed: Vec<(String, Tensor)>) -> Result<Vec<Tensor>> {
        let (input_names, input_values): (Vec<CString>, Vec<DynValue>) = input_feed
            .into_iter()
            .map(|(name, tensor)| Ok((CString::new(name)?, tensor.into_value()?)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let output_names = self
            .outputs
            .iter()
            .map(|node| CString::new(node.name.as_str()))
            .collect::<Result<Vec<_>, _>>()?;
        let input_name_ptrs: Vec<*const c_char> =
            input_names.iter().map(|name| name.as_ptr()).collect();
        let input_value_ptrs: Vec<*const ort::sys::OrtValue> =
            input_values.iter().map(AsPointer::ptr).collect();
        let output_name_ptrs: Vec<*const c_char> =
            output_names.iter().map(|name| name.as_ptr()).collect();
        let mut output_value_ptrs: Vec<*mut ort::sys::OrtValue> =
            vec![ptr::null_mut(); output_names.len()];
        unsafe {
            ort::Error::result_from_status((ort::api().Run)(
                self.session.ptr().cast_mut(),
                ptr::null(),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_value_ptrs.as_mut_ptr(),
            ))?;
        }
        let outputs = output_value_ptrs
            .into_iter()
            .map(|value| {
                let value = NonNull::new(value).context("ONNX Runtime returned no output")?;
                Ok(unsafe { DynValue::from_ptr(value, Some(self.session.allocator())) })
            })
            .collect::<Result<Vec<_>>>()?;
        self.outputs
            .iter()
            .zip(&outputs)
            .map(|(node, value)| {
                Ok(match node.ty.as_str() {
                    "tensor(float)" => Tensor::F32(value.try_extract_array::<f32>()?.to_owned()),
                    "tensor(float16)" => Tensor::F16(value.try_extract_array::<f16>()?.to_owned()),
                    "tensor(uint8)" => Tensor::U8(value.try_extract_array::<u8>()?.to_owned()),
                    "tensor(int32)" => Tensor::I32(value.try_extract_array::<i32>()?.to_owned()),
                    "tensor(int64)" => Tensor::I64(value.try_extract_array::<i64>()?.to_owned()),
                    other => bail!("unsupported output type {other}"),
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub kind: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GraphSpec {
    pub model_path: PathBuf,
    pub pins: Vec<(String, i64)>,
    pub overrides: Vec<(String, i64)>,
    pub providers: Vec<String>,
    pub disabled_optimizers: Vec<String>,
    pub threads: usize,
}

impl GraphSpec {
    pub fn provider(&self) -> &str {
        &self.providers[0]
    }

    pub fn directory(&self) -> Result<PathBuf> {
        let provider = self
            .provider()
            .strip_suffix("ExecutionProvider")
            .unwrap_or(self.provider())
            .to_lowercase();
        let mut directory = self
            .model_path
            .parent()
            .unwrap_or(Path::new(""))
            .join(provider);
        if let Some(device) = self.device()? {
            directory.push(device.kind);
        }
        directory.push(label(&self.pins));
        Ok(directory)
    }

    pub fn device(&self) -> Result<Option<Device>> {
        Ok(match self.provider() {
            "CPUExecutionProvider" => Some(Device {
                kind: machine().to_owned(),
                version: cpu()?.to_owned(),
            }),
            _ => None,
        })
    }

    pub fn manifest(&self) -> Result<PathBuf> {
        Ok(self.directory()?.join("manifest.json"))
    }

    pub fn plan(&self) -> RewritePlan {
        let target = PLANNED_AS
            .iter()
            .find(|(provider, _)| *provider == self.provider())
            .map_or(self.provider(), |(_, target)| target);
        plan_rewrites(&RewriteContext {
            target: target.to_owned(),
            ort_version: onnxruntime_version()
                .split('.')
                .take(3)
                .filter_map(|piece| piece.parse().ok())
                .collect(),
        })
    }

    pub fn half(&self) -> bool {
        !(settings().legacy_models() || self.provider() == "CPUExecutionProvider")
    }

    pub fn facts(&self) -> Result<Facts> {
        let rewrite = (!settings().legacy_models()).then(|| self.plan().digest);
        Ok(Facts {
            half: self.half(),
            onnxruntime: onnxruntime_version().to_owned(),
            options: self
                .provider_options()
                .swap_remove(0)
                .into_iter()
                .filter(|(key, _)| !key.starts_with("device"))
                .collect(),
            overrides: self.overrides.clone(),
            disabled_optimizers: self.disabled_optimizers.clone(),
            rewrite,
            device: self.device()?.map(|device| device.version),
        })
    }

    pub fn provider_options(&self) -> Vec<Map<String, Value>> {
        self.providers
            .iter()
            .map(|provider| {
                let mut options = Map::new();
                if provider == "CPUExecutionProvider" {
                    options.insert(
                        "arena_extend_strategy".to_owned(),
                        Value::from("kSameAsRequested"),
                    );
                }
                options
            })
            .collect()
    }

    pub fn session(&self, graph: &Path, sess_options: Option<SessionBuilder>) -> Result<Session> {
        let builder = match sess_options {
            Some(builder) => builder,
            None => self.sess_options()?,
        };
        let provider_options = self.provider_options();
        debug!(
            "Setting execution provider options to [{}]",
            provider_options
                .iter()
                .map(py_dict)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut builder = builder
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers([CPU::default()
                .with_arena_allocator(settings().model_arena)
                .build()])
            .map_err(ort::Error::<()>::from)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(ort::Error::<()>::from)?;
        if !self.disabled_optimizers.is_empty() {
            builder = builder
                .with_disabled_optimizers(self.disabled_optimizers.join(";"))
                .map_err(ort::Error::<()>::from)?;
        }
        Ok(builder.commit_from_file(graph)?)
    }

    pub fn sess_options(&self) -> Result<SessionBuilder> {
        let settings = settings();
        let mut builder = Session::builder(&ENVIRONMENT)?
            .with_flush_to_zero()
            .map_err(ort::Error::<()>::from)?;

        let cpu = self.provider() == "CPUExecutionProvider";
        let mut inter_op_num_threads = 0;
        if settings.model_inter_op_threads > 0 {
            inter_op_num_threads = settings.model_inter_op_threads as usize;
        } else if settings.model_inter_op_threads == 0 && cpu {
            inter_op_num_threads = 1;
        }
        if inter_op_num_threads > 0 {
            builder = builder
                .with_inter_threads(inter_op_num_threads)
                .map_err(ort::Error::<()>::from)?;
        }

        let mut intra_op_num_threads = 0;
        if settings.model_intra_op_threads > 0 {
            intra_op_num_threads = settings.model_intra_op_threads as usize;
        } else if settings.model_intra_op_threads == 0 && cpu {
            intra_op_num_threads = self.threads;
        }
        if intra_op_num_threads > 0 {
            builder = builder
                .with_intra_threads(intra_op_num_threads)
                .map_err(ort::Error::<()>::from)?;
        }

        let parallel = inter_op_num_threads > 1;
        if parallel {
            builder = builder
                .with_parallel_execution(true)
                .map_err(ort::Error::<()>::from)?;
        }

        for (dimension, size) in &self.overrides {
            builder = builder
                .with_dimension_override(dimension, *size)
                .map_err(ort::Error::<()>::from)?;
        }

        debug!(
            "Setting execution_mode to {}",
            if parallel {
                "ORT_PARALLEL"
            } else {
                "ORT_SEQUENTIAL"
            }
        );
        debug!("Setting inter_op_num_threads to {inter_op_num_threads}");
        debug!("Setting intra_op_num_threads to {intra_op_num_threads}");
        Ok(builder)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Facts {
    pub onnxruntime: String,
    pub options: Map<String, Value>,
    pub overrides: Vec<(String, i64)>,
    pub disabled_optimizers: Vec<String>,
    #[serde(default)]
    pub rewrite: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub half: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub facts: Facts,
    pub graph: String,
}

pub fn fresh(spec: &GraphSpec) -> Result<Option<PathBuf>> {
    let Ok(bytes) = std::fs::read(spec.manifest()?) else {
        return Ok(None);
    };
    let Ok(manifest) = serde_json::from_slice::<Manifest>(&bytes) else {
        return Ok(None);
    };
    let parent = spec.model_path.parent().unwrap_or(Path::new(""));
    Ok((manifest.facts == spec.facts()?).then(|| parent.join(manifest.graph)))
}

pub fn prepared(spec: &GraphSpec) -> Result<PathBuf> {
    if let Some(graph) = fresh(spec)? {
        return Ok(graph);
    }
    info!(
        "Preparing {} for {} {}",
        spec.model_path.display(),
        spec.provider(),
        pins_repr(&spec.pins)
    );
    let mut child = Command::new(std::env::current_exe()?)
        .arg("prepare")
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .context("the preparing child has no stdin")?
        .write_all(&serde_json::to_vec(spec)?)?;
    let status = child.wait()?;
    if let Some(graph) = fresh(spec)? {
        return Ok(graph);
    }
    if status.code() == Some(UNREADABLE) {
        return Err(ort::Error::new_with_code(
            ort::ErrorCode::InvalidProtobuf,
            format!("{} could not be read", spec.model_path.display()),
        )
        .into());
    }
    bail!(
        "Preparing {} failed; the output above says why",
        spec.model_path.display()
    )
}

pub fn machine() -> &'static str {
    static MACHINE: LazyLock<String> = LazyLock::new(|| unsafe {
        let mut name: libc::utsname = std::mem::zeroed();
        if libc::uname(&mut name) != 0 {
            return String::new();
        }
        CStr::from_ptr(name.machine.as_ptr())
            .to_string_lossy()
            .into_owned()
    });
    &MACHINE
}

fn cpu() -> Result<&'static str> {
    static CPU: OnceLock<String> = OnceLock::new();
    if let Some(cpu) = CPU.get() {
        return Ok(cpu);
    }
    let cpu = match std::fs::read_to_string("/proc/cpuinfo") {
        Err(_) => machine().to_owned(),
        Ok(cpuinfo) => cpuinfo
            .lines()
            .find(|line| line.starts_with("flags") || line.starts_with("Features"))
            .context("/proc/cpuinfo lists no flags")?
            .to_owned(),
    };
    Ok(CPU.get_or_init(|| cpu))
}

pub fn onnxruntime_version() -> &'static str {
    static VERSION: LazyLock<String> = LazyLock::new(|| unsafe {
        let base = ort::sys::OrtGetApiBase();
        CStr::from_ptr(((*base).GetVersionString)())
            .to_string_lossy()
            .into_owned()
    });
    &VERSION
}

fn pins_repr(pins: &[(String, i64)]) -> String {
    format!(
        "{{{}}}",
        pins.iter()
            .map(|(name, size)| format!("'{name}': {size}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn py_dict(options: &Map<String, Value>) -> String {
    format!(
        "{{{}}}",
        options
            .iter()
            .map(|(key, value)| match value {
                Value::String(text) => format!("'{key}': '{text}'"),
                other => format!("'{key}': {other}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub struct OrtSession {
    pub model_path: PathBuf,
    pub policy: ShapePolicy,
    pub providers: Vec<String>,
    disabled_optimizers: Vec<String>,
    threads: usize,
    dynamic: bool,
    pub shapes: Vec<Shape>,
    pub batches: Vec<i64>,
    graphs: Mutex<HashMap<Shape, Arc<OrtGraph>>>,
    locks: Mutex<HashMap<Shape, Arc<Mutex<()>>>>,
}

impl OrtSession {
    pub fn new(
        model_path: &Path,
        policy: ShapePolicy,
        providers: Option<Vec<String>>,
        threads: usize,
    ) -> Result<Self> {
        let providers = providers.unwrap_or_else(providers_default);
        info!(
            "Setting execution providers to {}, in descending order of preference",
            py_list(&providers)
        );
        let disabled_optimizers = disabled_optimizers_default(&providers);
        debug!(
            "Setting disabled_optimizers to {}",
            py_list(&disabled_optimizers)
        );
        let dynamic = DYNAMIC_PROVIDERS.contains(&providers[0].as_str());
        let shapes = if dynamic {
            vec![policy.pinned()]
        } else {
            policy.dims.clone()
        };
        let mut batches: Vec<i64> = shapes.iter().map(|shape| shape.batch).collect();
        batches.sort_unstable_by(|a, b| b.cmp(a));
        batches.dedup();
        let session = Self {
            model_path: model_path.to_path_buf(),
            policy,
            providers,
            disabled_optimizers,
            threads,
            dynamic,
            shapes,
            batches,
            graphs: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
        };
        if session.dynamic || session.policy.dims.len() == 1 {
            session.for_shape(session.policy.dims[0])?;
        }
        Ok(session)
    }

    pub fn for_shape(&self, shape: Shape) -> Result<Arc<OrtGraph>> {
        let shape = if self.dynamic { self.shapes[0] } else { shape };
        if let Some(graph) = self
            .graphs
            .lock()
            .map_err(|_| anyhow!("graph lock poisoned"))?
            .get(&shape)
        {
            return Ok(graph.clone());
        }
        if !self.dynamic && !self.policy.dims.contains(&shape) {
            bail!(
                "Asked for {}, which is none of {}",
                shape_repr(&shape),
                shape_list(&self.policy.dims)
            );
        }
        let lock = self
            .locks
            .lock()
            .map_err(|_| anyhow!("lock poisoned"))?
            .entry(shape)
            .or_default()
            .clone();
        let _guard = lock.lock().map_err(|_| anyhow!("lock poisoned"))?;
        if let Some(graph) = self
            .graphs
            .lock()
            .map_err(|_| anyhow!("graph lock poisoned"))?
            .get(&shape)
        {
            return Ok(graph.clone());
        }
        debug!("Building a graph for {}", shape_repr(&shape));
        let spec = GraphSpec {
            model_path: self.model_path.clone(),
            pins: shape
                .pins()
                .into_iter()
                .map(|(name, size)| (name.to_owned(), size))
                .collect(),
            overrides: overrides(&self.policy, &shape.pins()),
            providers: self.providers.clone(),
            disabled_optimizers: self.disabled_optimizers.clone(),
            threads: self.threads,
        };
        let graph = Arc::new(OrtGraph::new(&spec)?);
        self.graphs
            .lock()
            .map_err(|_| anyhow!("graph lock poisoned"))?
            .insert(shape, graph.clone());
        Ok(graph)
    }

    pub fn warm(&self) -> Result<()> {
        let shapes: Vec<Shape> = if self.dynamic {
            self.policy.dims[..1].to_vec()
        } else {
            self.policy.dims.clone()
        };
        for shape in shapes {
            let graph = self.for_shape(shape)?;
            let sizes: HashMap<String, i64> =
                overrides(&self.policy, &shape.pins()).into_iter().collect();
            let feed = graph
                .get_inputs()
                .iter()
                .map(|node| {
                    let dims = node
                        .shape
                        .iter()
                        .map(|dim| match dim {
                            Dim::Size(size) => *size as usize,
                            Dim::Symbol(symbol) => sizes.get(symbol).copied().unwrap_or(1) as usize,
                        })
                        .collect();
                    Ok((node.name.clone(), Tensor::zeros(&node.ty, dims)?))
                })
                .collect::<Result<Vec<_>>>()?;
            graph.infer(feed)?;
        }
        Ok(())
    }
}

pub fn overrides(policy: &ShapePolicy, pins: &[(&'static str, i64)]) -> Vec<(String, i64)> {
    let order: Vec<&str> = policy.dims[0]
        .pins()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let mut named = Vec::new();
    for (dimension, size) in pins {
        let position = order
            .iter()
            .position(|name| name == dimension)
            .unwrap_or_default();
        named.push(((*dimension).to_owned(), *size));
        named.push((format!("DynamicDimension.{position}"), *size));
        named.push((if position == 0 { "None" } else { "?" }.to_owned(), *size));
    }
    named
}

pub fn shape_repr(shape: &Shape) -> String {
    let optional =
        |value: Option<i64>| value.map_or_else(|| "None".to_owned(), |value| value.to_string());
    format!(
        "Shape(batch={}, height={}, width={})",
        shape.batch,
        optional(shape.height),
        optional(shape.width)
    )
}

fn shape_list(shapes: &[Shape]) -> String {
    format!(
        "[{}]",
        shapes.iter().map(shape_repr).collect::<Vec<_>>().join(", ")
    )
}

fn py_list(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|item| format!("'{item}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn providers_default() -> Vec<String> {
    let available = ["CPUExecutionProvider"];
    debug!(
        "Available ORT providers: {{{}}}",
        available
            .iter()
            .map(|item| format!("'{item}'"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    SUPPORTED_PROVIDERS
        .iter()
        .filter(|provider| available.contains(provider))
        .map(|provider| (*provider).to_owned())
        .collect()
}

fn disabled_optimizers_default(providers: &[String]) -> Vec<String> {
    let mut disabled_optimizers = Vec::new();
    if matches!(machine(), "arm64" | "aarch64") {
        disabled_optimizers.push("ConvAddActivationFusion".to_owned());
    }
    if providers.iter().any(|provider| {
        provider == "CoreMLExecutionProvider" || provider == "CUDAExecutionProvider"
    }) {
        disabled_optimizers.push("MatMulAddFusion".to_owned());
    }
    disabled_optimizers
}

pub fn flush_denormals() {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    unsafe {
        let mut csr: u32 = 0;
        std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack));
        csr |= 0x8040;
        std::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack, readonly));
    }
}
