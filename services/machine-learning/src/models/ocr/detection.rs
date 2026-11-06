use anyhow::Context;
use image::RgbImage;
use ndarray::{Array2, Array4, Axis, Ix2, s};

use super::legacy;
use super::postprocess::DBPostProcess;
use crate::immich_model::constants::ocr_canvases;
use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::transforms::widen;
use crate::numpy::argsort_stable;
use crate::pil::{self, Resampling};
use crate::schemas::{
    ModelFormat, ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output, Shape,
    TextDetectionOutput,
};
use crate::sessions::ort::{OrtGraph, OrtSession, Tensor};
use crate::sessions::policy::ShapePolicy;
use std::sync::Arc;

type Prepared = (Arc<OrtGraph>, Tensor, (usize, usize));

pub struct TextDetector {
    shape_policy: ShapePolicy,
    scale: f32,
    postprocess: DBPostProcess,
}

impl TextDetector {
    pub fn new(max_resolution: i64) -> Self {
        let size = Self::canvas_size(max_resolution);
        let canvases = ocr_canvases(size)
            .into_iter()
            .map(|(height, width)| Shape {
                batch: 1,
                height: Some(height),
                width: Some(width),
            })
            .collect();
        Self {
            shape_policy: ShapePolicy::with_label(canvases, format!("res{size}")),
            scale: (1.0f64 / 127.5) as f32,
            postprocess: DBPostProcess::new(0.3, 1000, 1.6, true).expect("dilation kernel"),
        }
    }

    pub fn graph(options: &crate::schemas::TextDetectionOptions) -> i64 {
        Self::canvas_size(options.max_resolution)
    }

    fn canvas_size(max_resolution: i64) -> i64 {
        (max_resolution as f64 / 32.0).ceil() as i64 * 32
    }

    fn round32(value: f64) -> i64 {
        ((value / 32.0).round_ties_even() * 32.0) as i64
    }

    fn canvas(shapes: &[Shape], height: i64, width: i64) -> (i64, i64) {
        let fit = |canvas: (i64, i64)| {
            (canvas.0 as f64 / height as f64)
                .min(canvas.1 as f64 / width as f64)
                .min(1.0)
        };
        shapes
            .iter()
            .filter_map(|shape| Some((shape.height?, shape.width?)))
            .min_by(|a, b| {
                (-fit(*a), a.0 * a.1)
                    .partial_cmp(&(-fit(*b), b.0 * b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((height, width))
    }

    fn transform(
        &self,
        session: &OrtSession,
        image: &RgbImage,
        max_resolution: i64,
    ) -> Result<Prepared, MlError> {
        let (width, height) = (i64::from(image.width()), i64::from(image.height()));
        let ratio = (max_resolution as f64 / height.min(width) as f64).min(1.0);
        let content_h = Self::round32(height as f64 * ratio).max(32);
        let content_w = Self::round32(width as f64 * ratio).max(32);
        let canvas = Self::canvas(&session.shapes, content_h, content_w);
        let graph = session.for_shape(Shape {
            batch: 1,
            height: Some(canvas.0),
            width: Some(canvas.1),
        })?;
        let scale = (canvas.0 as f64 / content_h as f64)
            .min(canvas.1 as f64 / content_w as f64)
            .min(1.0);
        let content_h = Self::round32(content_h as f64 * scale)
            .max(32)
            .min(canvas.0);
        let content_w = Self::round32(content_w as f64 * scale)
            .max(32)
            .min(canvas.1);
        let resized = pil::resize(
            image,
            content_w as u32,
            content_h as u32,
            Resampling::Lanczos,
        );
        let (content_h, content_w) = (content_h as usize, content_w as usize);
        let (height, width) = (canvas.0 as usize, canvas.1 as usize);
        let content = (content_h, content_w);
        let array = resized.as_raw();
        if graph.normalizes_input {
            let mut frame = Array4::<u8>::zeros((1, height, width, 3));
            for y in 0..content_h {
                for x in 0..content_w {
                    for c in 0..3 {
                        frame[[0, y, x, c]] = array[(y * content_w + x) * 3 + c];
                    }
                }
            }
            return Ok((graph, Tensor::U8(frame.into_dyn()), content));
        }
        let mut out = Array4::<f32>::from_elem((1, 3, height, width), -1.0);
        for channel in 0..3 {
            for y in 0..content_h {
                for x in 0..content_w {
                    let value = f32::from(array[(y * content_w + x) * 3 + channel]) * self.scale;
                    out[[0, 2 - channel, y, x]] = value - 1.0;
                }
            }
        }
        Ok((graph, Tensor::F32(out.into_dyn()), content))
    }

    pub fn reading_order(boxes: &[[[f32; 2]; 4]]) -> Vec<usize> {
        let tops: Vec<f32> = boxes.iter().map(|quad| quad[0][1]).collect();
        let y_order = argsort_stable(&tops);
        let mut lines = vec![0i32; boxes.len()];
        for i in 1..boxes.len() {
            lines[i] = lines[i - 1] + i32::from(tops[y_order[i]] - tops[y_order[i - 1]] >= 10.0);
        }
        let keys: Vec<f64> = y_order
            .iter()
            .zip(lines.iter())
            .map(|(index, line)| f64::from(*line) * 1e6 + f64::from(boxes[*index][0][0]))
            .collect();
        argsort_stable(&keys)
            .into_iter()
            .map(|position| y_order[position])
            .collect()
    }

    fn empty() -> Output {
        Output::TextDetection(TextDetectionOutput::default())
    }
}

impl Kind for TextDetector {
    type Loaded = OrtSession;
    const NAME: &'static str = "TextDetector";
    const IDENTITY: ModelIdentity = (ModelType::Detection, ModelTask::Ocr);
    const SOURCES: &'static [ModelSource] = &[ModelSource::Paddle];

    fn shape_policy(&self) -> &ShapePolicy {
        &self.shape_policy
    }

    fn model_format_default(&self) -> ModelFormat {
        legacy::model_format_default()
    }

    fn download(&self, model: &InferenceModel<Self>) -> Result<(), MlError> {
        legacy::download(model)
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
        let Options::TextDetection(options) = options else {
            return Err(anyhow::anyhow!("text detection needs text detection options").into());
        };
        let image = inputs[0].image()?;
        if image.width() < 32 || image.height() < 32 {
            return Ok(Self::empty());
        }
        let (graph, frame, content) = self.transform(session, image, options.max_resolution)?;
        let input = graph
            .get_inputs()
            .first()
            .context("graph has no inputs")?
            .name
            .clone();
        let output = graph
            .run(vec![(input, frame)])?
            .into_iter()
            .next()
            .context("graph has no outputs")?;
        let mut probs = widen(output)?.index_axis_move(Axis(0), 0);
        if probs.ndim() == 3 {
            probs = probs.index_axis_move(Axis(0), 0);
        }
        let probs: Array2<f32> = probs
            .into_dimensionality::<Ix2>()
            .map_err(anyhow::Error::from)?;
        let probs = probs.slice(s![..content.0, ..content.1]).to_owned();
        let (boxes, scores) = self.postprocess.call(
            &probs,
            (image.height() as usize, image.width() as usize),
            options.min_score,
        )?;
        if boxes.is_empty() {
            return Ok(Self::empty());
        }
        let order = Self::reading_order(&boxes);
        Ok(Output::TextDetection(TextDetectionOutput {
            boxes: order.iter().map(|index| boxes[*index]).collect(),
            scores: order.iter().map(|index| scores[*index]).collect(),
        }))
    }
}
