use std::path::PathBuf;

use anyhow::{Context, anyhow};
use image::RgbImage;
use ndarray::{Axis, Ix1};
use serde_json::Value;
use tracing::debug;

use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::transforms::{
    crop_pil, normalize, resize_pil, serialize_np_array, to_numpy, widen,
};
use crate::pil::{self, Resampling};
use crate::schemas::{ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output};
use crate::sessions::ort::{OrtGraph, OrtSession, Tensor};
use crate::sessions::policy::ShapePolicy;

pub struct VisualState {
    session: OrtSession,
    size: u32,
    resampling: Resampling,
    mean: [f32; 3],
    std: [f32; 3],
    resize_mode: String,
    preprocess_cfg_path: PathBuf,
}

pub struct OpenClipVisualEncoder {
    shape_policy: ShapePolicy,
}

impl OpenClipVisualEncoder {
    pub fn new() -> Self {
        Self {
            shape_policy: ShapePolicy::default(),
        }
    }

    fn transform(
        state: &VisualState,
        session: &OrtGraph,
        image: &RgbImage,
    ) -> Result<Vec<(String, Tensor)>, MlError> {
        let image = Self::resize(state, image)?;
        if session.normalizes_input {
            let (width, height) = image.dimensions();
            let rgb = ndarray::Array::from_shape_vec(
                (1, height as usize, width as usize, 3),
                image.into_raw(),
            )
            .map_err(anyhow::Error::from)?;
            return Ok(vec![("image".to_owned(), Tensor::U8(rgb.into_dyn()))]);
        }
        let mut array = to_numpy(&image);
        normalize(&mut array, &state.mean, &state.std);
        let chw = array
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_owned()
            .insert_axis(Axis(0));
        Ok(vec![("image".to_owned(), Tensor::F32(chw.into_dyn()))])
    }

    fn resize(state: &VisualState, image: &RgbImage) -> Result<RgbImage, MlError> {
        match state.resize_mode.as_str() {
            "squash" => Ok(pil::resize(image, state.size, state.size, state.resampling)),
            "shortest" => Ok(crop_pil(
                &resize_pil(image, state.size, state.resampling),
                state.size,
            )),
            mode => Err(anyhow!(
                "Unsupported resize_mode '{mode}' in {}",
                state.preprocess_cfg_path.display()
            )
            .into()),
        }
    }
}

impl Default for OpenClipVisualEncoder {
    fn default() -> Self {
        Self::new()
    }
}

fn floats(value: &Value, field: &str) -> anyhow::Result<[f32; 3]> {
    let values: Vec<f32> = value
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("preprocess config has no {field}"))?
        .iter()
        .map(|value| value.as_f64().map(|value| value as f32))
        .collect::<Option<_>>()
        .with_context(|| format!("preprocess config {field} is not numeric"))?;
    values
        .try_into()
        .map_err(|_| anyhow!("preprocess config {field} needs three values"))
}

impl Kind for OpenClipVisualEncoder {
    type Loaded = VisualState;
    const NAME: &'static str = "OpenClipVisualEncoder";
    const IDENTITY: ModelIdentity = (ModelType::Visual, ModelTask::Search);
    const SOURCES: &'static [ModelSource] = &[ModelSource::Openclip, ModelSource::Mclip];

    fn shape_policy(&self) -> &ShapePolicy {
        &self.shape_policy
    }

    fn load(&self, model: &InferenceModel<Self>) -> Result<VisualState, MlError> {
        let preprocess_cfg_path = model.model_dir().join("preprocess_cfg.json");
        debug!(
            "Loading visual preprocessing config for CLIP model '{}'",
            model.model_name
        );
        let preprocess_cfg: Value = serde_json::from_reader(
            std::fs::File::open(&preprocess_cfg_path).map_err(anyhow::Error::from)?,
        )
        .map_err(anyhow::Error::from)?;
        debug!(
            "Loaded visual preprocessing config for CLIP model '{}'",
            model.model_name
        );
        let size = match &preprocess_cfg["size"] {
            Value::Array(sizes) => sizes.first().and_then(Value::as_u64),
            other => other.as_u64(),
        }
        .context("preprocess config has no size")? as u32;
        let resampling = Resampling::parse(
            preprocess_cfg["interpolation"]
                .as_str()
                .context("preprocess config has no interpolation")?,
        )
        .map_err(|name| anyhow!("KeyError: {name}"))?;
        let mean = floats(&preprocess_cfg, "mean")?;
        let std = floats(&preprocess_cfg, "std")?;
        let resize_mode = preprocess_cfg
            .get("resize_mode")
            .and_then(Value::as_str)
            .unwrap_or("shortest")
            .to_owned();
        let session = model.make_session()?;
        Ok(VisualState {
            session,
            size,
            resampling,
            mean,
            std,
            resize_mode,
            preprocess_cfg_path,
        })
    }

    fn session(loaded: &VisualState) -> &OrtSession {
        &loaded.session
    }

    fn predict(
        &self,
        _: &InferenceModel<Self>,
        state: &VisualState,
        inputs: &[Input],
        _: &Options,
    ) -> Result<Output, MlError> {
        let image = inputs[0].image()?;
        let session = state.session.for_shape(self.shape_policy.dims[0])?;
        let feed = Self::transform(state, &session, image)?;
        let output = session
            .run(feed)?
            .into_iter()
            .next()
            .context("graph has no outputs")?;
        let res = widen(output)?;
        let embedding = res
            .index_axis(Axis(0), 0)
            .into_dimensionality::<Ix1>()
            .map_err(anyhow::Error::from)?;
        if embedding.is_empty() {
            return Err(anyhow!("empty embedding").into());
        }
        Ok(Output::Embedding(serialize_np_array(&embedding.to_vec())))
    }
}
