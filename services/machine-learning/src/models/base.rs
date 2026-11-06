use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::anyhow;
use image::RgbImage;
use tracing::{debug, info, warn};

use super::clip::textual::{MClipTextualEncoder, OpenClipTextualEncoder};
use super::clip::visual::OpenClipVisualEncoder;
use super::constants::get_model_source;
use super::facial_recognition::detection::FaceDetector;
use super::facial_recognition::recognition::FaceRecognizer;
use super::ocr::detection::TextDetector;
use super::ocr::recognition::TextRecognizer;
use crate::config::{clean_name, settings};
use crate::huggingface_hub::snapshot_download;
use crate::schemas::{
    ModelFormat, ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output, py_repr_str,
};
use crate::sessions::ort::OrtSession;
use crate::sessions::policy::ShapePolicy;

#[derive(Clone)]
pub enum Input {
    Image(Arc<RgbImage>),
    Text(Arc<str>),
    Output(Arc<Output>),
}

impl Input {
    pub fn image(&self) -> anyhow::Result<&RgbImage> {
        match self {
            Self::Image(image) => Ok(image),
            _ => Err(anyhow!("expected an image")),
        }
    }

    pub fn text(&self) -> anyhow::Result<&str> {
        match self {
            Self::Text(text) => Ok(text),
            _ => Err(anyhow!("expected text")),
        }
    }

    pub fn output(&self) -> anyhow::Result<&Output> {
        match self {
            Self::Output(output) => Ok(output),
            _ => Err(anyhow!("expected the output of another model")),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlError {
    #[error("{detail}")]
    Http { status: u16, detail: String },
    #[error("{0}")]
    FileNotFound(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl MlError {
    pub fn http(status: u16, detail: impl Into<String>) -> Self {
        Self::Http {
            status,
            detail: detail.into(),
        }
    }

    pub fn is_os_error(&self) -> bool {
        match self {
            Self::FileNotFound(_) => true,
            Self::Http { .. } => false,
            Self::Other(error) => error
                .chain()
                .any(|cause| cause.is::<std::io::Error>() || cause.is::<reqwest::Error>()),
        }
    }

    pub fn is_model_file_error(&self) -> bool {
        match self {
            Self::Other(error) => error.chain().any(|cause| {
                cause.downcast_ref::<ort::Error>().is_some_and(|error| {
                    matches!(
                        error.code(),
                        ort::ErrorCode::NoSuchFile | ort::ErrorCode::InvalidProtobuf
                    )
                })
            }),
            _ => false,
        }
    }
}

pub trait Kind: Send + Sync + Sized + 'static {
    type Loaded: Send + Sync + 'static;
    const NAME: &'static str;
    const IDENTITY: ModelIdentity;
    const DEPENDS: &'static [ModelIdentity] = &[];
    const SOURCES: &'static [ModelSource];
    const THREADS: usize = 2;

    fn shape_policy(&self) -> &ShapePolicy;

    fn model_format_default(&self) -> ModelFormat {
        ModelFormat::Onnx
    }

    fn download(&self, model: &InferenceModel<Self>) -> Result<(), MlError> {
        model.download_snapshot()
    }

    fn load(&self, model: &InferenceModel<Self>) -> Result<Self::Loaded, MlError>;

    fn session(loaded: &Self::Loaded) -> &OrtSession;

    fn predict(
        &self,
        model: &InferenceModel<Self>,
        loaded: &Self::Loaded,
        inputs: &[Input],
        options: &Options,
    ) -> Result<Output, MlError>;
}

pub struct InferenceModel<K: Kind> {
    pub kind: K,
    pub model_name: String,
    pub cache_dir: PathBuf,
    model_format: Mutex<ModelFormat>,
    load_lock: Mutex<()>,
    state: RwLock<Option<Arc<K::Loaded>>>,
    load_attempts: AtomicUsize,
}

impl<K: Kind> InferenceModel<K> {
    pub fn new(kind: K, model_name: &str) -> Self {
        let model_name = clean_name(model_name);
        let cache_dir = Self::cache_dir_default(&model_name);
        let model_format = kind.model_format_default();
        debug!("Setting model format to {model_format}");
        Self {
            kind,
            model_name,
            cache_dir,
            model_format: Mutex::new(model_format),
            load_lock: Mutex::new(()),
            state: RwLock::new(None),
            load_attempts: AtomicUsize::new(0),
        }
    }

    fn cache_dir_default(model_name: &str) -> PathBuf {
        let settings = settings();
        let cache_dir = settings
            .cache_folder
            .join(K::IDENTITY.1.value())
            .join(model_name);
        if settings.legacy_models() {
            cache_dir
        } else {
            cache_dir.join(&settings.model_revision)
        }
    }

    pub fn model_type(&self) -> ModelType {
        K::IDENTITY.0
    }

    pub fn model_task(&self) -> ModelTask {
        K::IDENTITY.1
    }

    pub fn format(&self) -> ModelFormat {
        *self.model_format.lock().expect("model format lock")
    }

    pub fn model_dir(&self) -> PathBuf {
        self.cache_dir.join(self.model_type().value())
    }

    pub fn model_path(&self) -> PathBuf {
        self.model_dir().join(format!("model.{}", self.format()))
    }

    pub fn cached(&self) -> bool {
        self.model_path().is_file()
    }

    pub fn download_snapshot(&self) -> Result<(), MlError> {
        if self.cached() {
            return Ok(());
        }
        let settings = settings();
        info!(
            "Downloading {} model '{}' to {}. This may take a while.",
            self.model_type().value().replace('-', " "),
            self.model_name,
            self.model_dir().display()
        );
        let ignore_patterns: &[&str] = match self.format() {
            ModelFormat::Onnx => &["*.armnn", "*.rknn"],
            ModelFormat::Armnn => &["*.rknn"],
            ModelFormat::Rknn => &["*.armnn"],
        };
        snapshot_download(
            &format!(
                "{}/{}",
                settings.model_organization.value(),
                self.model_name
            ),
            &settings.model_revision,
            &self.cache_dir,
            ignore_patterns,
        )?;
        if !self.cached() {
            return Err(MlError::FileNotFound(format!(
                "Model file not found: {}",
                self.model_path().display()
            )));
        }
        Ok(())
    }

    pub fn make_session(&self) -> Result<OrtSession, MlError> {
        match self.format() {
            ModelFormat::Onnx => Ok(OrtSession::new(
                &self.model_path(),
                self.kind.shape_policy().clone(),
                None,
                K::THREADS,
            )?),
            other => Err(MlError::FileNotFound(format!(
                "{} is not supported by this build",
                other.value().to_uppercase()
            ))),
        }
    }

    fn loaded_state(&self) -> Option<Arc<K::Loaded>> {
        self.state.read().expect("model state lock").clone()
    }
}

pub trait AnyModel: Send + Sync {
    fn class_name(&self) -> &'static str;
    fn identity(&self) -> ModelIdentity;
    fn depends(&self) -> &'static [ModelIdentity];
    fn model_name(&self) -> &str;
    fn model_type(&self) -> ModelType;
    fn model_format(&self) -> ModelFormat;
    fn set_model_format(&self, model_format: ModelFormat);
    fn loaded(&self) -> bool;
    fn load_attempts(&self) -> usize;
    fn load(&self) -> Result<(), MlError>;
    fn unload(&self);
    fn build(&self) -> Result<(), MlError>;
    fn predict(&self, inputs: &[Input], options: &Options) -> Result<Output, MlError>;
    fn clear_cache(&self) -> Result<(), MlError>;
}

impl<K: Kind> AnyModel for InferenceModel<K> {
    fn class_name(&self) -> &'static str {
        K::NAME
    }

    fn identity(&self) -> ModelIdentity {
        K::IDENTITY
    }

    fn depends(&self) -> &'static [ModelIdentity] {
        K::DEPENDS
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn model_type(&self) -> ModelType {
        K::IDENTITY.0
    }

    fn model_format(&self) -> ModelFormat {
        self.format()
    }

    fn set_model_format(&self, model_format: ModelFormat) {
        debug!("Setting model format to {model_format}");
        *self.model_format.lock().expect("model format lock") = model_format;
    }

    fn loaded(&self) -> bool {
        self.state.read().expect("model state lock").is_some()
    }

    fn load_attempts(&self) -> usize {
        self.load_attempts.load(Ordering::SeqCst)
    }

    fn load(&self) -> Result<(), MlError> {
        let _guard = self.load_lock.lock().expect("model load lock");
        if self.loaded() {
            return Ok(());
        }
        let attempts = self.load_attempts.fetch_add(1, Ordering::SeqCst) + 1;
        self.kind.download(self)?;
        let attempt = if attempts > 1 {
            format!("Attempt #{attempts} to load")
        } else {
            "Loading".to_owned()
        };
        info!(
            "{attempt} {} model '{}' to memory",
            self.model_type().value().replace('-', " "),
            self.model_name
        );
        let loaded = self.kind.load(self)?;
        *self.state.write().expect("model state lock") = Some(Arc::new(loaded));
        Ok(())
    }

    fn unload(&self) {
        let _guard = self.load_lock.lock().expect("model load lock");
        *self.state.write().expect("model state lock") = None;
    }

    fn build(&self) -> Result<(), MlError> {
        self.load()?;
        let loaded = self
            .loaded_state()
            .ok_or_else(|| anyhow!("model was unloaded"))?;
        K::session(&loaded).warm()?;
        Ok(())
    }

    fn predict(&self, inputs: &[Input], options: &Options) -> Result<Output, MlError> {
        self.load()?;
        let loaded = self
            .loaded_state()
            .ok_or_else(|| anyhow!("model was unloaded"))?;
        self.kind.predict(self, &loaded, inputs, options)
    }

    fn clear_cache(&self) -> Result<(), MlError> {
        if !self.cache_dir.exists() {
            warn!(
                "Attempted to clear cache for model '{}', but cache directory does not exist",
                self.model_name
            );
            return Ok(());
        }
        if self.cache_dir.is_dir() {
            info!("Cleared cache directory for model '{}'.", self.model_name);
            std::fs::remove_dir_all(&self.cache_dir).map_err(anyhow::Error::from)?;
        } else {
            warn!(
                "Encountered file instead of directory at cache path for '{}'. Removing file and replacing with a directory.",
                self.model_name
            );
            std::fs::remove_file(&self.cache_dir).map_err(anyhow::Error::from)?;
        }
        std::fs::create_dir_all(&self.cache_dir).map_err(anyhow::Error::from)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelClass {
    OpenClipVisualEncoder,
    OpenClipTextualEncoder,
    MClipTextualEncoder,
    FaceDetector,
    FaceRecognizer,
    TextDetector,
    TextRecognizer,
}

impl ModelClass {
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenClipVisualEncoder => <OpenClipVisualEncoder as Kind>::NAME,
            Self::OpenClipTextualEncoder => <OpenClipTextualEncoder as Kind>::NAME,
            Self::MClipTextualEncoder => <MClipTextualEncoder as Kind>::NAME,
            Self::FaceDetector => <FaceDetector as Kind>::NAME,
            Self::FaceRecognizer => <FaceRecognizer as Kind>::NAME,
            Self::TextDetector => <TextDetector as Kind>::NAME,
            Self::TextRecognizer => <TextRecognizer as Kind>::NAME,
        }
    }

    pub fn identity(self) -> ModelIdentity {
        match self {
            Self::OpenClipVisualEncoder => <OpenClipVisualEncoder as Kind>::IDENTITY,
            Self::OpenClipTextualEncoder => <OpenClipTextualEncoder as Kind>::IDENTITY,
            Self::MClipTextualEncoder => <MClipTextualEncoder as Kind>::IDENTITY,
            Self::FaceDetector => <FaceDetector as Kind>::IDENTITY,
            Self::FaceRecognizer => <FaceRecognizer as Kind>::IDENTITY,
            Self::TextDetector => <TextDetector as Kind>::IDENTITY,
            Self::TextRecognizer => <TextRecognizer as Kind>::IDENTITY,
        }
    }

    pub fn sources(self) -> &'static [ModelSource] {
        match self {
            Self::OpenClipVisualEncoder => <OpenClipVisualEncoder as Kind>::SOURCES,
            Self::OpenClipTextualEncoder => <OpenClipTextualEncoder as Kind>::SOURCES,
            Self::MClipTextualEncoder => <MClipTextualEncoder as Kind>::SOURCES,
            Self::FaceDetector => <FaceDetector as Kind>::SOURCES,
            Self::FaceRecognizer => <FaceRecognizer as Kind>::SOURCES,
            Self::TextDetector => <TextDetector as Kind>::SOURCES,
            Self::TextRecognizer => <TextRecognizer as Kind>::SOURCES,
        }
    }

    pub fn graph(self, options: &Options) -> Option<i64> {
        match (self, options) {
            (Self::TextDetector, Options::TextDetection(options)) => {
                Some(TextDetector::graph(options))
            }
            _ => None,
        }
    }

    pub fn create(self, model_name: &str, options: &Options) -> Arc<dyn AnyModel> {
        match self {
            Self::OpenClipVisualEncoder => Arc::new(InferenceModel::new(
                OpenClipVisualEncoder::new(),
                model_name,
            )),
            Self::OpenClipTextualEncoder => Arc::new(InferenceModel::new(
                OpenClipTextualEncoder::new(),
                model_name,
            )),
            Self::MClipTextualEncoder => {
                Arc::new(InferenceModel::new(MClipTextualEncoder::new(), model_name))
            }
            Self::FaceDetector => Arc::new(InferenceModel::new(FaceDetector::new(), model_name)),
            Self::FaceRecognizer => {
                Arc::new(InferenceModel::new(FaceRecognizer::new(), model_name))
            }
            Self::TextDetector => {
                let max_resolution = match options {
                    Options::TextDetection(options) => options.max_resolution,
                    _ => crate::schemas::TextDetectionOptions::default_max_resolution(),
                };
                Arc::new(InferenceModel::new(
                    TextDetector::new(max_resolution),
                    model_name,
                ))
            }
            Self::TextRecognizer => {
                Arc::new(InferenceModel::new(TextRecognizer::new(), model_name))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferenceEntry {
    pub model: ModelClass,
    pub name: String,
    pub options: Options,
}

impl InferenceEntry {
    pub fn new(model: ModelClass, name: String, options: Options) -> anyhow::Result<Self> {
        if !get_model_source(&name).is_some_and(|source| model.sources().contains(&source)) {
            anyhow::bail!("'{name}' is not a model {} runs", model.name());
        }
        Ok(Self {
            model,
            name,
            options,
        })
    }
}

impl fmt::Display for InferenceEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(name={}, options={})",
            self.model.name(),
            py_repr_str(&self.name),
            self.options
        )
    }
}
