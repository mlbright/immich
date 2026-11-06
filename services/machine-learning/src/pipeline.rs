use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::models::base::{InferenceEntry, ModelClass};
use crate::models::constants::get_model_source;
use crate::schemas::{
    FaceDetectionOptions, FaceRecognitionOptions, ModelSource, Options, TextDetectionOptions,
    TextRecognitionOptions, TextualOptions, VisualOptions,
};

pub fn textual(model_name: &str) -> ModelClass {
    if get_model_source(model_name) == Some(ModelSource::Mclip) {
        ModelClass::MClipTextualEncoder
    } else {
        ModelClass::OpenClipTextualEncoder
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, bound = "O: DeserializeOwned + Default")]
pub struct Slot<O> {
    #[serde(rename = "modelName")]
    pub model_name: String,
    #[serde(default)]
    pub options: O,
}

impl<O: Clone> Slot<O> {
    pub fn new(model_name: impl Into<String>, options: O) -> Self {
        Self {
            model_name: model_name.into(),
            options,
        }
    }

    fn entry(&self, model: ModelClass, wrap: fn(O) -> Options) -> anyhow::Result<InferenceEntry> {
        InferenceEntry::new(model, self.model_name.clone(), wrap(self.options.clone()))
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Clip {
    pub visual: Option<Slot<VisualOptions>>,
    pub textual: Option<Slot<TextualOptions>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FacialRecognition {
    pub detection: Option<Slot<FaceDetectionOptions>>,
    pub recognition: Option<Slot<FaceRecognitionOptions>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Ocr {
    pub detection: Option<Slot<TextDetectionOptions>>,
    pub recognition: Option<Slot<TextRecognitionOptions>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PipelineRequest {
    pub clip: Clip,
    #[serde(rename = "facial-recognition")]
    pub facial_recognition: FacialRecognition,
    pub ocr: Ocr,
}

impl PipelineRequest {
    pub fn entries(&self) -> anyhow::Result<Vec<InferenceEntry>> {
        let mut entries = Vec::new();
        if let Some(visual) = &self.clip.visual {
            entries.push(visual.entry(ModelClass::OpenClipVisualEncoder, Options::Visual)?);
        }
        if let Some(text) = &self.clip.textual {
            entries.push(text.entry(textual(&text.model_name), Options::Textual)?);
        }
        if let Some(detection) = &self.facial_recognition.detection {
            entries.push(detection.entry(ModelClass::FaceDetector, Options::FaceDetection)?);
        }
        if let Some(recognition) = &self.facial_recognition.recognition {
            entries.push(recognition.entry(ModelClass::FaceRecognizer, Options::FaceRecognition)?);
        }
        if let Some(boxes) = &self.ocr.detection {
            entries.push(boxes.entry(ModelClass::TextDetector, Options::TextDetection)?);
        }
        if let Some(reading) = &self.ocr.recognition {
            entries.push(reading.entry(ModelClass::TextRecognizer, Options::TextRecognition)?);
        }
        Ok(entries)
    }
}
