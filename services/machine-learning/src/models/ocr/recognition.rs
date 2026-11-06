use std::sync::OnceLock;

use anyhow::{Context, anyhow};
use image::RgbImage;
use nalgebra::{SMatrix, SVector};
use ndarray::{Array4, ArrayD};

use super::ctc::{CtcDecoder, greedy_for, py_splitlines};
use super::legacy;
use crate::config::settings;
use crate::immich_model::constants::OCR_RECOGNITION_WIDTHS;
use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::transforms::{cv_resize, widen};
use crate::pil;
use crate::schemas::{
    ModelFormat, ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output, Shape,
    TextDetectionOutput, TextRecognitionOutput,
};
use crate::sessions::ort::{OrtGraph, OrtSession, Tensor};
use crate::sessions::policy::{ShapePolicy, batches, runs};

pub const REC_HEIGHT: usize = 48;

pub type CropGeometry = (Vec<i32>, Vec<i32>, Vec<[f64; 8]>);

pub struct RecognizerState {
    session: OrtSession,
    widths: Vec<i64>,
    decoder: OnceLock<CtcDecoder>,
}

pub struct TextRecognizer {
    shape_policy: ShapePolicy,
    scale: f32,
}

fn norm(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    ((0.0 + dx * dx) + dy * dy).sqrt()
}

impl TextRecognizer {
    pub fn new() -> Self {
        let sizes = batches(settings().max_batch_size.ocr);
        let dims = OCR_RECOGNITION_WIDTHS
            .iter()
            .flat_map(|width| {
                sizes.iter().map(move |batch| Shape {
                    batch: *batch,
                    height: None,
                    width: Some(*width),
                })
            })
            .collect();
        Self {
            shape_policy: ShapePolicy::new(dims),
            scale: (1.0f64 / 127.5) as f32,
        }
    }

    fn decoder(model: &InferenceModel<Self>, graph: &OrtGraph) -> anyhow::Result<CtcDecoder> {
        let greedy = greedy_for(
            graph
                .get_outputs()
                .first()
                .map(|node| node.name.as_str())
                .unwrap_or_default(),
        );
        match graph.get_metadata().get("character") {
            Some(character) => Ok(CtcDecoder::new(py_splitlines(character), greedy)),
            None => CtcDecoder::from_file(&model.model_dir().join("charset.txt"), greedy),
        }
    }

    fn width(widths: &[i64], width: i64) -> i64 {
        if widths.is_empty() {
            let first = OCR_RECOGNITION_WIDTHS[0];
            let last = OCR_RECOGNITION_WIDTHS[OCR_RECOGNITION_WIDTHS.len() - 1];
            return last.min(first.max((width as f64 * 1.25).ceil() as i64));
        }
        let position = widths.partition_point(|candidate| *candidate < width);
        widths[position.min(widths.len() - 1)]
    }

    pub fn crop_geometry(boxes: &[[[f32; 2]; 4]]) -> anyhow::Result<CropGeometry> {
        let mut widths = Vec::with_capacity(boxes.len());
        let mut heights = Vec::with_capacity(boxes.len());
        let mut coeffs = Vec::with_capacity(boxes.len());
        for quad in boxes {
            let width = norm(quad[1], quad[0]).max(norm(quad[2], quad[3])) as i32;
            let height = norm(quad[0], quad[3]).max(norm(quad[1], quad[2])) as i32;
            let src = [
                [0.0f32, 0.0],
                [width as f32, 0.0],
                [width as f32, height as f32],
                [0.0, height as f32],
            ];
            coeffs.push(Self::get_perspective_transform(&src, quad)?);
            widths.push(width);
            heights.push(height);
        }
        Ok((widths, heights, coeffs))
    }

    fn get_perspective_transform(
        src: &[[f32; 2]; 4],
        dst: &[[f32; 2]; 4],
    ) -> anyhow::Result<[f64; 8]> {
        let mut a = SMatrix::<f32, 8, 8>::zeros();
        let mut b = SVector::<f32, 8>::zeros();
        for point in 0..4 {
            let (x, y) = (src[point][0], src[point][1]);
            let (u, v) = (dst[point][0], dst[point][1]);
            let even = point * 2;
            a[(even, 0)] = x;
            a[(even, 1)] = y;
            a[(even, 2)] = 1.0;
            a[(even, 6)] = -u * x;
            a[(even, 7)] = -u * y;
            b[even] = u;
            let odd = even + 1;
            a[(odd, 3)] = x;
            a[(odd, 4)] = y;
            a[(odd, 5)] = 1.0;
            a[(odd, 6)] = -v * x;
            a[(odd, 7)] = -v * y;
            b[odd] = v;
        }
        let solution = a.lu().solve(&b).ok_or_else(|| anyhow!("Singular matrix"))?;
        Ok(std::array::from_fn(|index| f64::from(solution[index])))
    }

    pub fn crop(
        image: &RgbImage,
        coeffs: &[f64; 8],
        width: i32,
        height: i32,
        upright: bool,
    ) -> RgbImage {
        let crop = pil::transform_perspective_bicubic(
            image,
            width.max(0) as u32,
            height.max(0) as u32,
            coeffs,
        );
        if upright { pil::rotate_90(&crop) } else { crop }
    }

    fn empty() -> Output {
        Output::TextRecognition(TextRecognitionOutput {
            empty: true,
            ..TextRecognitionOutput::default()
        })
    }
}

impl Default for TextRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Kind for TextRecognizer {
    type Loaded = RecognizerState;
    const NAME: &'static str = "TextRecognizer";
    const IDENTITY: ModelIdentity = (ModelType::Recognition, ModelTask::Ocr);
    const DEPENDS: &'static [ModelIdentity] = &[(ModelType::Detection, ModelTask::Ocr)];
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

    fn load(&self, model: &InferenceModel<Self>) -> Result<RecognizerState, MlError> {
        let session = model.make_session()?;
        let mut widths: Vec<i64> = session
            .shapes
            .iter()
            .filter_map(|shape| shape.width)
            .collect();
        widths.sort_unstable();
        widths.dedup();
        Ok(RecognizerState {
            session,
            widths,
            decoder: OnceLock::new(),
        })
    }

    fn session(loaded: &RecognizerState) -> &OrtSession {
        &loaded.session
    }

    fn predict(
        &self,
        model: &InferenceModel<Self>,
        state: &RecognizerState,
        inputs: &[Input],
        options: &Options,
    ) -> Result<Output, MlError> {
        let Options::TextRecognition(options) = options else {
            return Err(anyhow!("text recognition needs text recognition options").into());
        };
        let image = inputs[0].image()?;
        let Output::TextDetection(texts) = inputs[1].output()? else {
            return Err(anyhow!("text recognition needs text detection output").into());
        };
        let TextDetectionOutput {
            boxes,
            scores: box_scores,
        } = texts;
        if boxes.is_empty() {
            return Ok(Self::empty());
        }

        let (widths, heights, coeffs) = Self::crop_geometry(boxes)?;
        let upright: Vec<bool> = widths
            .iter()
            .zip(&heights)
            .map(|(width, height)| height * 2 >= width * 3)
            .collect();
        let ratios: Vec<f64> = (0..boxes.len())
            .map(|i| {
                if upright[i] {
                    f64::from(heights[i]) / f64::from(widths[i])
                } else {
                    f64::from(widths[i]) / f64::from(heights[i])
                }
            })
            .collect();
        let mut order: Vec<usize> = (0..boxes.len()).collect();
        order.sort_by(|a, b| {
            ratios[*a]
                .partial_cmp(&ratios[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut text_list = vec![String::new(); order.len()];
        let mut score_list = vec![0.0f32; order.len()];
        let sized = runs(order.len(), &state.session.batches);
        let mut start = 0;
        for size in sized {
            let chunk = &order[start..start + size];
            start += size;
            let width = Self::width(
                &state.widths,
                (REC_HEIGHT as f64 * ratios[chunk[chunk.len() - 1]]) as i64,
            );
            let graph = state.session.for_shape(Shape {
                batch: chunk.len() as i64,
                height: None,
                width: Some(width),
            })?;
            let raw = graph.normalizes_input;
            let width = width as usize;
            let mut images_u8 = Array4::<u8>::zeros(if raw {
                (chunk.len(), REC_HEIGHT, width, 3)
            } else {
                (0, 0, 0, 0)
            });
            let mut images = Array4::<f32>::zeros(if raw {
                (0, 0, 0, 0)
            } else {
                (chunk.len(), 3, REC_HEIGHT, width)
            });
            for (i, index) in chunk.iter().enumerate() {
                let crop = Self::crop(
                    image,
                    &coeffs[*index],
                    widths[*index],
                    heights[*index],
                    upright[*index],
                );
                let resized_w =
                    1.max(width.min((REC_HEIGHT as f64 * ratios[*index]).ceil() as usize));
                let resized = cv_resize(&crop, resized_w as u32, REC_HEIGHT as u32)?;
                let pixels = resized.as_raw();
                for y in 0..REC_HEIGHT {
                    for x in 0..width {
                        for c in 0..3 {
                            if raw {
                                images_u8[[i, y, x, c]] = if x < resized_w {
                                    pixels[(y * resized_w + x) * 3 + c]
                                } else {
                                    127
                                };
                            } else if x < resized_w {
                                images[[i, 2 - c, y, x]] =
                                    f32::from(pixels[(y * resized_w + x) * 3 + c]) * self.scale
                                        - 1.0;
                            }
                        }
                    }
                }
            }
            let input = graph
                .get_inputs()
                .first()
                .context("graph has no inputs")?
                .name
                .clone();
            let frames = if raw {
                Tensor::U8(images_u8.into_dyn())
            } else {
                Tensor::F32(images.into_dyn())
            };
            let outputs: Vec<ArrayD<f32>> = graph
                .run(vec![(input, frames)])?
                .into_iter()
                .map(widen)
                .collect::<anyhow::Result<_>>()?;
            let decoder = match state.decoder.get() {
                Some(decoder) => decoder,
                None => {
                    let decoder = Self::decoder(model, &graph)?;
                    state.decoder.get_or_init(|| decoder)
                }
            };
            let (chunk_texts, chunk_scores) = decoder.call(&outputs)?;
            for ((index, text), score) in chunk.iter().zip(chunk_texts).zip(chunk_scores) {
                text_list[*index] = text;
                score_list[*index] = score;
            }
        }

        let image_width = image.width() as f32;
        let image_height = image.height() as f32;
        let min_score = options.min_score as f32;
        let mut output = TextRecognitionOutput::default();
        for (index, quad) in boxes.iter().enumerate() {
            if score_list[index] > min_score {
                for point in quad {
                    output.r#box.push(point[0] / image_width);
                    output.r#box.push(point[1] / image_height);
                }
                output.text.push(text_list[index].clone());
                output.box_score.push(box_scores[index]);
                output.text_score.push(score_list[index]);
            }
        }
        Ok(Output::TextRecognition(output))
    }
}
