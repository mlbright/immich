use anyhow::Context;
use ndarray::{Array3, Axis};

use super::_ops::{decode_scrfd, nms};
use crate::immich_model::constants::FACE_DETECTION_SIZE as DET_SIZE;
use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::transforms::{letterbox, widen};
use crate::schemas::{
    FaceDetectionOutput, ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output, Shape,
};
use crate::sessions::ort::{OrtSession, Tensor};
use crate::sessions::policy::ShapePolicy;

pub struct FaceDetector {
    shape_policy: ShapePolicy,
}

impl FaceDetector {
    pub fn new() -> Self {
        Self {
            shape_policy: ShapePolicy::new(vec![Shape {
                batch: 1,
                height: Some(DET_SIZE),
                width: Some(DET_SIZE),
            }]),
        }
    }
}

impl Default for FaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Kind for FaceDetector {
    type Loaded = OrtSession;
    const NAME: &'static str = "FaceDetector";
    const IDENTITY: ModelIdentity = (ModelType::Detection, ModelTask::FacialRecognition);
    const SOURCES: &'static [ModelSource] = &[ModelSource::Insightface];

    fn shape_policy(&self) -> &ShapePolicy {
        &self.shape_policy
    }

    fn load(&self, model: &InferenceModel<Self>) -> Result<OrtSession, MlError> {
        model.make_session()
    }

    fn session(loaded: &OrtSession) -> &OrtSession {
        loaded
    }

    fn predict(
        &self,
        _: &InferenceModel<Self>,
        session: &OrtSession,
        inputs: &[Input],
        options: &Options,
    ) -> Result<Output, MlError> {
        let min_score = match options {
            Options::FaceDetection(options) => options.min_score as f32,
            _ => 0.7,
        };
        let session = session.for_shape(self.shape_policy.dims[0])?;
        let (canvas, scale) = letterbox(inputs[0].image()?, DET_SIZE as u32)?;
        let size = DET_SIZE as usize;
        let hwc = Array3::from_shape_vec((size, size, 3), canvas.into_raw())
            .map_err(anyhow::Error::from)?;
        let blob = if session.normalizes_input {
            Tensor::U8(hwc.insert_axis(Axis(0)).into_dyn())
        } else {
            let factor = (1.0f64 / 128.0) as f32;
            let shift = (127.5f64 / 128.0) as f32;
            let normalized = hwc.mapv(|value| f32::from(value) * factor - shift);
            Tensor::F32(
                normalized
                    .permuted_axes([2, 0, 1])
                    .as_standard_layout()
                    .into_owned()
                    .insert_axis(Axis(0))
                    .into_dyn(),
            )
        };
        let input = session
            .get_inputs()
            .first()
            .context("graph has no inputs")?
            .name
            .clone();
        let heads = session
            .run(vec![(input, blob)])?
            .into_iter()
            .map(widen)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let decoded = decode_scrfd(heads, DET_SIZE);

        let scale = scale as f32;
        let mut scores = Vec::new();
        let mut boxes = Vec::new();
        let mut kps = Vec::new();
        for (index, score) in decoded.scores.iter().enumerate() {
            if *score >= min_score {
                scores.push(*score);
                boxes.push(decoded.boxes[index].map(|value| value / scale));
                kps.push(decoded.kps[index].map(|value| value / scale));
            }
        }
        let keep = nms(&boxes, &scores, 0.4);
        Ok(Output::FaceDetection(FaceDetectionOutput {
            boxes: keep
                .iter()
                .map(|index| boxes[*index].map(f32::round_ties_even))
                .collect(),
            scores: keep.iter().map(|index| scores[*index]).collect(),
            landmarks: keep
                .iter()
                .map(|index| {
                    let points = kps[*index];
                    std::array::from_fn(|point| [points[point * 2], points[point * 2 + 1]])
                })
                .collect(),
        }))
    }
}
