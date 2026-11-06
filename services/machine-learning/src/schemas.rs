use std::fmt;

use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! str_enum {
    ($name:ident { $($variant:ident = $member:literal : $value:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub const fn value(self) -> &'static str {
                match self {
                    $(Self::$variant => $value),+
                }
            }

            pub const fn member(self) -> &'static str {
                match self {
                    $(Self::$variant => $member),+
                }
            }

            pub fn parse(value: &str) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub fn repr(self) -> String {
                format!("<{}.{}: '{}'>", stringify!($name), self.member(), self.value())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.value())
            }
        }
    };
}

str_enum!(ModelTask {
    FacialRecognition = "FACIAL_RECOGNITION": "facial-recognition",
    Search = "SEARCH": "clip",
    Ocr = "OCR": "ocr",
});

str_enum!(ModelType {
    Detection = "DETECTION": "detection",
    Recognition = "RECOGNITION": "recognition",
    Textual = "TEXTUAL": "textual",
    Visual = "VISUAL": "visual",
});

str_enum!(ModelFormat {
    Armnn = "ARMNN": "armnn",
    Onnx = "ONNX": "onnx",
    Rknn = "RKNN": "rknn",
});

str_enum!(ModelSource {
    Insightface = "INSIGHTFACE": "insightface",
    Mclip = "MCLIP": "mclip",
    Openclip = "OPENCLIP": "openclip",
    Paddle = "PADDLE": "paddle",
});

str_enum!(ModelOrganization {
    App = "APP": "immich-app",
    Testing = "TESTING": "immich-testing",
});

pub type ModelIdentity = (ModelType, ModelTask);

pub fn identity_repr((model_type, task): ModelIdentity) -> String {
    format!("({}, {})", model_type.repr(), task.repr())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    pub batch: i64,
    pub height: Option<i64>,
    pub width: Option<i64>,
}

impl Shape {
    pub const fn new(batch: i64) -> Self {
        Self {
            batch,
            height: None,
            width: None,
        }
    }

    pub fn pins(&self) -> Vec<(&'static str, i64)> {
        let mut pins = vec![("batch", self.batch)];
        if let Some(height) = self.height {
            pins.push(("height", height));
        }
        if let Some(width) = self.width {
            pins.push(("width", width));
        }
        pins
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FaceDetectionOutput {
    pub boxes: Vec<[f32; 4]>,
    pub scores: Vec<f32>,
    pub landmarks: Vec<[[f32; 2]; 5]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DetectedFace {
    #[serde(rename = "boundingBox")]
    pub bounding_box: BoundingBox,
    pub embedding: String,
    pub score: f32,
}

pub type FacialRecognitionOutput = Vec<DetectedFace>;

fn python_float<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    use serde::de::Error;
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| D::Error::custom("Input should be a valid number")),
        serde_json::Value::String(text) => text
            .trim()
            .parse()
            .map_err(|_| D::Error::custom("Input should be a valid number")),
        serde_json::Value::Bool(flag) => Ok(f64::from(u8::from(flag))),
        _ => Err(D::Error::custom("Input should be a valid number")),
    }
}

fn python_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    use serde::de::Error;
    let invalid = || D::Error::custom("Input should be a valid integer");
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(number) => match number.as_i64() {
            Some(value) => Ok(value),
            None => number
                .as_f64()
                .filter(|value| value.fract() == 0.0)
                .map(|value| value as i64)
                .ok_or_else(invalid),
        },
        serde_json::Value::String(text) => text.trim().parse().map_err(|_| invalid()),
        serde_json::Value::Bool(flag) => Ok(i64::from(flag)),
        _ => Err(invalid()),
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct VisualOptions {}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct TextualOptions {
    #[serde(default)]
    pub language: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FaceDetectionOptions {
    #[serde(
        rename = "minScore",
        default = "FaceDetectionOptions::default_min_score",
        deserialize_with = "python_float"
    )]
    pub min_score: f64,
}

impl FaceDetectionOptions {
    fn default_min_score() -> f64 {
        0.7
    }
}

impl Default for FaceDetectionOptions {
    fn default() -> Self {
        Self {
            min_score: Self::default_min_score(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct FaceRecognitionOptions {}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TextDetectionOptions {
    #[serde(
        rename = "maxResolution",
        default = "TextDetectionOptions::default_max_resolution",
        deserialize_with = "python_int"
    )]
    pub max_resolution: i64,
    #[serde(
        rename = "minScore",
        default = "TextDetectionOptions::default_min_score",
        deserialize_with = "python_float"
    )]
    pub min_score: f64,
}

impl TextDetectionOptions {
    pub const fn default_max_resolution() -> i64 {
        736
    }

    fn default_min_score() -> f64 {
        0.5
    }
}

impl Default for TextDetectionOptions {
    fn default() -> Self {
        Self {
            max_resolution: Self::default_max_resolution(),
            min_score: Self::default_min_score(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TextRecognitionOptions {
    #[serde(
        rename = "minScore",
        default = "TextRecognitionOptions::default_min_score",
        deserialize_with = "python_float"
    )]
    pub min_score: f64,
}

impl TextRecognitionOptions {
    fn default_min_score() -> f64 {
        0.9
    }
}

impl Default for TextRecognitionOptions {
    fn default() -> Self {
        Self {
            min_score: Self::default_min_score(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Options {
    Visual(VisualOptions),
    Textual(TextualOptions),
    FaceDetection(FaceDetectionOptions),
    FaceRecognition(FaceRecognitionOptions),
    TextDetection(TextDetectionOptions),
    TextRecognition(TextRecognitionOptions),
}

pub fn py_repr_str(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(value.len() + 2);
    repr.push(quote);
    for c in value.chars() {
        match c {
            '\\' => repr.push_str("\\\\"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\t' => repr.push_str("\\t"),
            c if c == quote => {
                repr.push('\\');
                repr.push(c);
            }
            c => repr.push(c),
        }
    }
    repr.push(quote);
    repr
}

pub fn py_repr_float(value: f64) -> String {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 1e16 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

impl fmt::Display for Options {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Visual(_) => write!(f, "VisualOptions()"),
            Self::Textual(options) => match &options.language {
                Some(language) => write!(f, "TextualOptions(language={})", py_repr_str(language)),
                None => write!(f, "TextualOptions(language=None)"),
            },
            Self::FaceDetection(options) => write!(
                f,
                "FaceDetectionOptions(min_score={})",
                py_repr_float(options.min_score)
            ),
            Self::FaceRecognition(_) => write!(f, "FaceRecognitionOptions()"),
            Self::TextDetection(options) => write!(
                f,
                "TextDetectionOptions(max_resolution={}, min_score={})",
                options.max_resolution,
                py_repr_float(options.min_score)
            ),
            Self::TextRecognition(options) => {
                write!(
                    f,
                    "TextRecognitionOptions(min_score={})",
                    py_repr_float(options.min_score)
                )
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextDetectionOutput {
    pub boxes: Vec<[[f32; 2]; 4]>,
    pub scores: Vec<f32>,
}

impl Serialize for TextDetectionOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("TextDetectionOutput", 2)?;
        state.serialize_field("boxes", &self.boxes)?;
        state.serialize_field("scores", &self.scores)?;
        state.end()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextRecognitionOutput {
    pub r#box: Vec<f32>,
    pub box_score: Vec<f32>,
    pub text: Vec<String>,
    pub text_score: Vec<f32>,
    pub empty: bool,
}

impl Serialize for TextRecognitionOutput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_map(Some(4))?;
        state.serialize_entry("box", &self.r#box)?;
        if self.empty {
            state.serialize_entry("boxScore", &self.box_score)?;
            state.serialize_entry("text", &self.text)?;
        } else {
            state.serialize_entry("text", &self.text)?;
            state.serialize_entry("boxScore", &self.box_score)?;
        }
        state.serialize_entry("textScore", &self.text_score)?;
        state.end()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Embedding(String),
    FaceDetection(FaceDetectionOutput),
    FacialRecognition(FacialRecognitionOutput),
    TextDetection(TextDetectionOutput),
    TextRecognition(TextRecognitionOutput),
}

impl Serialize for Output {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Embedding(embedding) => embedding.serialize(serializer),
            Self::FaceDetection(output) => output.serialize(serializer),
            Self::FacialRecognition(output) => output.serialize(serializer),
            Self::TextDetection(output) => output.serialize(serializer),
            Self::TextRecognition(output) => output.serialize(serializer),
        }
    }
}

pub struct InferenceResponse {
    pub entries: Vec<(ModelTask, std::sync::Arc<Output>)>,
    pub image_size: Option<(u32, u32)>,
}

impl Serialize for InferenceResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (task, output) in &self.entries {
            map.serialize_entry(task.value(), output.as_ref())?;
        }
        if let Some((height, width)) = self.image_size {
            map.serialize_entry("imageHeight", &height)?;
            map.serialize_entry("imageWidth", &width)?;
        }
        map.end()
    }
}
