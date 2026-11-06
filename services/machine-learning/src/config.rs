use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};
use tracing::Level;

use crate::schemas::ModelOrganization;

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ClipSettings {
    pub textual: Option<String>,
    pub visual: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct FacialRecognitionSettings {
    pub recognition: Option<String>,
    pub detection: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct OcrSettings {
    pub recognition: Option<String>,
    pub detection: Option<String>,
    #[serde(deserialize_with = "lenient::int")]
    pub max_resolution: i64,
}

impl Default for OcrSettings {
    fn default() -> Self {
        Self {
            recognition: None,
            detection: None,
            max_resolution: 736,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PreloadModelData {
    pub clip: ClipSettings,
    pub facial_recognition: FacialRecognitionSettings,
    pub ocr: OcrSettings,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct MaxBatchSize {
    #[serde(deserialize_with = "lenient::uint")]
    pub facial_recognition: usize,
    #[serde(deserialize_with = "lenient::uint")]
    pub ocr: usize,
}

impl Default for MaxBatchSize {
    fn default() -> Self {
        Self {
            facial_recognition: 4,
            ocr: 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub cache_folder: PathBuf,
    pub model_ttl: i64,
    pub model_ttl_poll_s: i64,
    pub workers: i64,
    pub worker_timeout: i64,
    pub http_keepalive_timeout_s: i64,
    pub test_full: bool,
    pub test_provider: Option<String>,
    pub request_threads: i64,
    pub model_inter_op_threads: i64,
    pub model_intra_op_threads: i64,
    pub model_arena: bool,
    pub ann: bool,
    pub ann_fp16_turbo: bool,
    pub ann_tuning_level: i64,
    pub rknn: bool,
    pub rknn_threads: Option<i64>,
    pub preload: Option<PreloadModelData>,
    pub max_batch_size: MaxBatchSize,
    pub model_organization: ModelOrganization,
    pub model_revision: String,
}

impl Settings {
    pub fn legacy_models(&self) -> bool {
        self.model_revision == "main"
    }

    pub fn device_id(&self) -> String {
        std::env::var("MACHINE_LEARNING_DEVICE_ID").unwrap_or_else(|_| "0".to_owned())
    }

    pub fn from_env() -> Result<Self> {
        let tree = Environment::read("machine_learning_");
        Ok(Self {
            cache_folder: match tree.string("cache_folder") {
                Some(folder) => PathBuf::from(folder),
                None => default_cache_folder(),
            },
            model_ttl: tree.int("model_ttl")?.unwrap_or(300),
            model_ttl_poll_s: tree.int("model_ttl_poll_s")?.unwrap_or(10),
            workers: tree.int("workers")?.unwrap_or(1),
            worker_timeout: tree.int("worker_timeout")?.unwrap_or_else(default_worker_timeout),
            http_keepalive_timeout_s: tree.int("http_keepalive_timeout_s")?.unwrap_or(2),
            test_full: tree.bool("test_full")?.unwrap_or(false),
            test_provider: tree.string("test_provider"),
            request_threads: tree.int("request_threads")?.unwrap_or_else(cpu_count),
            model_inter_op_threads: tree.int("model_inter_op_threads")?.unwrap_or(0),
            model_intra_op_threads: tree.int("model_intra_op_threads")?.unwrap_or(0),
            model_arena: tree.bool("model_arena")?.unwrap_or(true),
            ann: tree.bool("ann")?.unwrap_or(true),
            ann_fp16_turbo: tree.bool("ann_fp16_turbo")?.unwrap_or(false),
            ann_tuning_level: tree.int("ann_tuning_level")?.unwrap_or(2),
            rknn: tree.bool("rknn")?.unwrap_or(true),
            rknn_threads: tree.int("rknn_threads")?,
            preload: tree.model("preload")?,
            max_batch_size: tree.model("max_batch_size")?.unwrap_or_default(),
            model_organization: match tree.string("model_organization") {
                Some(organization) => ModelOrganization::parse(&organization)
                    .with_context(|| format!("model_organization: Input should be 'immich-app' or 'immich-testing', got {organization:?}"))?,
                None => ModelOrganization::App,
            },
            model_revision: tree.string("model_revision").unwrap_or_else(|| "main".to_owned()),
        })
    }
}

#[derive(Clone, Debug)]
pub struct NonPrefixedSettings {
    pub immich_host: String,
    pub immich_port: u16,
    pub immich_log_level: String,
    pub no_color: bool,
}

impl NonPrefixedSettings {
    pub fn from_env() -> Result<Self> {
        let tree = Environment::read("");
        Ok(Self {
            immich_host: tree
                .string("immich_host")
                .unwrap_or_else(|| "[::]".to_owned()),
            immich_port: match tree.int("immich_port")? {
                Some(port) => u16::try_from(port)
                    .with_context(|| format!("immich_port: {port} is not a valid port"))?,
                None => 3003,
            },
            immich_log_level: tree
                .string("immich_log_level")
                .unwrap_or_else(|| "info".to_owned()),
            no_color: tree.bool("no_color")?.unwrap_or(false),
        })
    }
}

fn default_cache_folder() -> PathBuf {
    let folder = std::env::home_dir()
        .unwrap_or_default()
        .join(".cache")
        .join("immich_ml");
    std::path::absolute(&folder).unwrap_or(folder)
}

fn default_worker_timeout() -> i64 {
    if std::env::var("DEVICE").as_deref() == Ok("rocm") {
        900
    } else {
        300
    }
}

fn cpu_count() -> i64 {
    let online = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if online > 0 { online as i64 } else { 4 }
}

pub fn clean_name(model_name: &str) -> String {
    model_name
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .chars()
        .filter(|c| *c != '.')
        .map(|c| if c == ':' || c == '\\' { '_' } else { c })
        .collect()
}

pub fn log_level(name: &str) -> Level {
    match name.to_lowercase().as_str() {
        "critical" | "error" => Level::ERROR,
        "warning" | "warn" => Level::WARN,
        "debug" | "verbose" => Level::DEBUG,
        _ => Level::INFO,
    }
}

struct Environment {
    values: BTreeMap<String, String>,
}

impl Environment {
    fn read(prefix: &str) -> Self {
        let values = std::env::vars_os()
            .filter_map(|(key, value)| {
                Some((
                    key.into_string().ok()?.to_lowercase(),
                    value.into_string().ok()?,
                ))
            })
            .filter_map(|(key, value)| Some((key.strip_prefix(prefix)?.to_owned(), value)))
            .collect();
        Self { values }
    }

    fn string(&self, field: &str) -> Option<String> {
        self.values.get(field).cloned()
    }

    fn int(&self, field: &str) -> Result<Option<i64>> {
        self.values
            .get(field)
            .map(|value| {
                value.trim().parse().with_context(|| {
                    format!("{field}: Input should be a valid integer, got {value:?}")
                })
            })
            .transpose()
    }

    fn bool(&self, field: &str) -> Result<Option<bool>> {
        self.values
            .get(field)
            .map(|value| match value.trim().to_lowercase().as_str() {
                "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
                "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
                _ => bail!("{field}: Input should be a valid boolean, got {value:?}"),
            })
            .transpose()
    }

    fn model<T: for<'de> Deserialize<'de>>(&self, field: &str) -> Result<Option<T>> {
        let nested_prefix = format!("{field}__");
        let mut tree = match self.values.get(field) {
            Some(value) => match serde_json::from_str::<Value>(value)
                .with_context(|| format!("{field}: error parsing value"))?
            {
                Value::Object(object) => Some(object),
                _ => bail!("{field}: Input should be an object"),
            },
            None => None,
        };
        for (key, value) in &self.values {
            let Some(path) = key.strip_prefix(&nested_prefix) else {
                continue;
            };
            let mut node = tree.get_or_insert_with(Map::new);
            let parts: Vec<&str> = path.split("__").collect();
            for part in &parts[..parts.len() - 1] {
                let child = node
                    .entry((*part).to_owned())
                    .or_insert_with(|| Value::Object(Map::new()));
                if !child.is_object() {
                    *child = Value::Object(Map::new());
                }
                node = child.as_object_mut().expect("just made an object");
            }
            let leaf = serde_json::from_str::<Value>(value)
                .ok()
                .filter(|parsed| parsed.is_object() || parsed.is_array())
                .unwrap_or_else(|| Value::String(value.clone()));
            node.insert(parts[parts.len() - 1].to_owned(), leaf);
        }
        tree.map(|object| {
            serde_json::from_value(Value::Object(object))
                .with_context(|| format!("{field}: invalid value"))
        })
        .transpose()
    }
}

mod lenient {
    use serde::{Deserialize, Deserializer, de::Error};
    use serde_json::Value;

    pub fn int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Number(number) => number
                .as_i64()
                .ok_or_else(|| D::Error::custom("Input should be a valid integer")),
            Value::String(text) => text
                .trim()
                .parse()
                .map_err(|_| D::Error::custom("Input should be a valid integer")),
            _ => Err(D::Error::custom("Input should be a valid integer")),
        }
    }

    pub fn uint<'de, D: Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
        let value = int(deserializer)?;
        usize::try_from(value)
            .map_err(|_| D::Error::custom("Input should be a non-negative integer"))
    }
}

pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| {
    Settings::from_env()
        .unwrap_or_else(|error| panic!("invalid machine learning settings: {error:#}"))
});

pub static NON_PREFIXED_SETTINGS: LazyLock<NonPrefixedSettings> = LazyLock::new(|| {
    NonPrefixedSettings::from_env().unwrap_or_else(|error| panic!("invalid settings: {error:#}"))
});

pub fn settings() -> &'static Settings {
    &SETTINGS
}
