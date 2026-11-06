use anyhow::{Context, anyhow};
use ndarray::{Array4, Axis, Ix2, concatenate, s};

use super::_ops::{ALIGNED_SIZE, align_face, image_mat};
use crate::config::settings;
use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::transforms::{serialize_np_array, widen};
use crate::schemas::{
    BoundingBox, DetectedFace, FaceDetectionOutput, ModelIdentity, ModelSource, ModelTask,
    ModelType, Options, Output, Shape,
};
use crate::sessions::ort::{OrtSession, Tensor};
use crate::sessions::policy::{ShapePolicy, batches, runs};

pub struct FaceRecognizer {
    shape_policy: ShapePolicy,
}

impl FaceRecognizer {
    pub fn new() -> Self {
        let sizes = batches(settings().max_batch_size.facial_recognition);
        Self {
            shape_policy: ShapePolicy::new(sizes.into_iter().map(Shape::new).collect()),
        }
    }

    fn predict_batch(
        session: &OrtSession,
        crops: &Array4<f32>,
        crops_u8: &Array4<u8>,
        raw: bool,
        sizes: &[usize],
    ) -> Result<ndarray::Array2<f32>, MlError> {
        let mut embeddings = Vec::new();
        let mut start = 0;
        for size in sizes {
            let graph = session.for_shape(Shape::new(*size as i64))?;
            let input = graph
                .get_inputs()
                .first()
                .context("graph has no inputs")?
                .name
                .clone();
            let batch = if raw {
                Tensor::U8(
                    crops_u8
                        .slice(s![start..start + size, .., .., ..])
                        .to_owned()
                        .into_dyn(),
                )
            } else {
                Tensor::F32(
                    crops
                        .slice(s![start..start + size, .., .., ..])
                        .to_owned()
                        .into_dyn(),
                )
            };
            let output = graph
                .run(vec![(input, batch)])?
                .into_iter()
                .next()
                .context("graph has no outputs")?;
            embeddings.push(
                widen(output)?
                    .into_dimensionality::<Ix2>()
                    .map_err(anyhow::Error::from)?,
            );
            start += size;
        }
        if embeddings.len() == 1 {
            return Ok(embeddings.pop().expect("one batch"));
        }
        let views: Vec<_> = embeddings
            .iter()
            .map(|embedding| embedding.view())
            .collect();
        Ok(concatenate(Axis(0), &views).map_err(anyhow::Error::from)?)
    }

    fn postprocess(
        faces: &FaceDetectionOutput,
        embeddings: &ndarray::Array2<f32>,
    ) -> Vec<DetectedFace> {
        faces
            .boxes
            .iter()
            .zip(embeddings.outer_iter())
            .zip(faces.scores.iter())
            .map(|((bounding_box, embedding), score)| DetectedFace {
                bounding_box: BoundingBox {
                    x1: bounding_box[0],
                    y1: bounding_box[1],
                    x2: bounding_box[2],
                    y2: bounding_box[3],
                },
                embedding: serialize_np_array(&embedding.to_vec()),
                score: *score,
            })
            .collect()
    }
}

impl Default for FaceRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Kind for FaceRecognizer {
    type Loaded = OrtSession;
    const NAME: &'static str = "FaceRecognizer";
    const IDENTITY: ModelIdentity = (ModelType::Recognition, ModelTask::FacialRecognition);
    const DEPENDS: &'static [ModelIdentity] =
        &[(ModelType::Detection, ModelTask::FacialRecognition)];
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
        _: &Options,
    ) -> Result<Output, MlError> {
        let Output::FaceDetection(faces) = inputs[1].output()? else {
            return Err(anyhow!("face recognition needs face detection output").into());
        };
        if faces.boxes.is_empty() {
            return Ok(Output::FacialRecognition(Vec::new()));
        }
        let image = image_mat(inputs[0].image()?)?;
        let sizes = runs(faces.landmarks.len(), &session.batches);
        let side = ALIGNED_SIZE as usize;
        let mut crops_u8 = Array4::<u8>::zeros((faces.landmarks.len(), side, side, 3));
        for (mut crop, kps) in crops_u8.outer_iter_mut().zip(faces.landmarks.iter()) {
            let aligned = align_face(&image, kps)?;
            for ((y, x, c), value) in crop.indexed_iter_mut() {
                *value = aligned.get_pixel(x as u32, y as u32)[c];
            }
        }
        let raw = session
            .for_shape(Shape::new(sizes[0] as i64))?
            .normalizes_input;
        let crops = if raw {
            Array4::<f32>::zeros((0, 3, side, side))
        } else {
            let factor = (1.0f64 / 127.5) as f32;
            let shift = 1.0f32;
            crops_u8
                .view()
                .permuted_axes([0, 3, 1, 2])
                .mapv(|value| f32::from(value) * factor - shift)
                .as_standard_layout()
                .into_owned()
        };
        let embeddings = Self::predict_batch(session, &crops, &crops_u8, raw, &sizes)?;
        Ok(Output::FacialRecognition(Self::postprocess(
            faces,
            &embeddings,
        )))
    }
}
