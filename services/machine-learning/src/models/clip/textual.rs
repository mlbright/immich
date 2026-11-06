use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, anyhow};
use ndarray::{Array2, Axis, Ix1};
use serde_json::Value;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};
use tracing::{debug, warn};

use crate::models::base::{InferenceModel, Input, Kind, MlError};
use crate::models::constants::flores200;
use crate::models::transforms::{clean_text, serialize_np_array, widen};
use crate::schemas::{ModelIdentity, ModelSource, ModelTask, ModelType, Options, Output};
use crate::sessions::ort::{OrtSession, Tensor};
use crate::sessions::policy::ShapePolicy;

pub struct TextualState {
    session: OrtSession,
    tokenizer: Tokenizer,
    canonicalize: bool,
    is_nllb: bool,
}

fn read_json(path: &Path) -> Result<Value, MlError> {
    let file = std::fs::File::open(path).map_err(anyhow::Error::from)?;
    Ok(serde_json::from_reader(file).map_err(anyhow::Error::from)?)
}

fn load_textual<K: Kind>(model: &InferenceModel<K>) -> Result<(OrtSession, Value), MlError> {
    let session = model.make_session()?;
    let graph = session.for_shape(model.kind.shape_policy().dims[0])?;
    if graph.get_metadata().contains_key("embedding") {
        return Err(anyhow!(
            "'{}' keeps its token table outside the graph, which this build does not read yet",
            model.model_name
        )
        .into());
    }
    debug!("Loading tokenizer for CLIP model '{}'", model.model_name);
    debug!("Loading model config for CLIP model '{}'", model.model_name);
    let model_cfg = read_json(&model.cache_dir.join("config.json"))?;
    debug!("Loaded model config for CLIP model '{}'", model.model_name);
    let text_cfg = model_cfg
        .get("text_cfg")
        .cloned()
        .context("KeyError: 'text_cfg'")?;
    Ok((session, text_cfg))
}

fn open_clip_tokenizer<K: Kind>(
    model: &InferenceModel<K>,
    text_cfg: &Value,
) -> Result<Tokenizer, MlError> {
    let context_length = text_cfg
        .get("context_length")
        .and_then(Value::as_u64)
        .unwrap_or(77) as usize;
    debug!(
        "Loading tokenizer config for CLIP model '{}'",
        model.model_name
    );
    let tokenizer_cfg = read_json(&model.model_dir().join("tokenizer_config.json"))?;
    debug!(
        "Loaded tokenizer config for CLIP model '{}'",
        model.model_name
    );
    let pad_token = tokenizer_cfg
        .get("pad_token")
        .and_then(Value::as_str)
        .context("KeyError: 'pad_token'")?
        .to_owned();
    let mut tokenizer = Tokenizer::from_file(model.model_dir().join("tokenizer.json"))
        .map_err(|error| anyhow!("{error}"))?;
    let pad_id = tokenizer
        .token_to_id(&pad_token)
        .ok_or_else(|| anyhow!("Pad token '{pad_token}' not found in tokenizer vocab"))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(context_length),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id,
        pad_type_id: 0,
        pad_token,
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            direction: TruncationDirection::Right,
            max_length: context_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }))
        .map_err(|error| anyhow!("{error}"))?;
    Ok(tokenizer)
}

fn textual_state<K: Kind>(model: &InferenceModel<K>) -> Result<TextualState, MlError> {
    let (session, text_cfg) = load_textual(model)?;
    let tokenizer = open_clip_tokenizer(model, &text_cfg)?;
    let canonicalize = text_cfg
        .get("tokenizer_kwargs")
        .and_then(Value::as_object)
        .is_some_and(|kwargs| kwargs.get("clean").and_then(Value::as_str) == Some("canonicalize"));
    let is_nllb = model.model_name.starts_with("nllb");
    debug!("Loaded tokenizer for CLIP model '{}'", model.model_name);
    Ok(TextualState {
        session,
        tokenizer,
        canonicalize,
        is_nllb,
    })
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<tokenizers::Encoding, MlError> {
    Ok(tokenizer
        .encode(text, true)
        .map_err(|error| anyhow!("{error}"))?)
}

fn row(values: &[u32]) -> Tensor {
    let values: Vec<i32> = values.iter().map(|value| *value as i32).collect();
    let length = values.len();
    Tensor::I32(
        Array2::from_shape_vec((1, length), values)
            .expect("one row")
            .into_dyn(),
    )
}

fn run_textual(
    state: &TextualState,
    policy: &ShapePolicy,
    mut tokens: HashMap<String, Tensor>,
) -> Result<Output, MlError> {
    let graph = state.session.for_shape(policy.dims[0])?;
    let feed = graph
        .get_inputs()
        .iter()
        .map(|node| {
            tokens
                .remove(&node.name)
                .map(|tensor| (node.name.clone(), tensor))
                .ok_or_else(|| anyhow!("KeyError: '{}'", node.name))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let output = graph
        .run(feed)?
        .into_iter()
        .next()
        .context("graph has no outputs")?;
    let res = widen(output)?;
    let embedding = res
        .index_axis(Axis(0), 0)
        .into_dimensionality::<Ix1>()
        .map_err(anyhow::Error::from)?;
    Ok(Output::Embedding(serialize_np_array(&embedding.to_vec())))
}

pub struct OpenClipTextualEncoder {
    shape_policy: ShapePolicy,
}

impl OpenClipTextualEncoder {
    pub fn new() -> Self {
        Self {
            shape_policy: ShapePolicy::default(),
        }
    }

    pub fn tokenize(
        state: &TextualState,
        text: &str,
        language: Option<&str>,
    ) -> Result<HashMap<String, Tensor>, MlError> {
        let mut text = clean_text(text, state.canonicalize);
        if state.is_nllb
            && let Some(language) = language
        {
            let flores_code = flores200(language)
                .or_else(|| flores200(language.split('-').next().unwrap_or_default()))
                .unwrap_or_else(|| {
                    warn!("Language '{language}' not found, defaulting to 'en'");
                    "eng_Latn"
                });
            text = format!("{flores_code}{text}");
        }
        let tokens = encode(&state.tokenizer, &text)?;
        Ok(HashMap::from([("text".to_owned(), row(tokens.get_ids()))]))
    }
}

impl Default for OpenClipTextualEncoder {
    fn default() -> Self {
        Self::new()
    }
}

fn language(options: &Options) -> Option<&str> {
    match options {
        Options::Textual(options) => options.language.as_deref(),
        _ => None,
    }
}

impl Kind for OpenClipTextualEncoder {
    type Loaded = TextualState;
    const NAME: &'static str = "OpenClipTextualEncoder";
    const IDENTITY: ModelIdentity = (ModelType::Textual, ModelTask::Search);
    const SOURCES: &'static [ModelSource] = &[ModelSource::Openclip];
    const THREADS: usize = 4;

    fn shape_policy(&self) -> &ShapePolicy {
        &self.shape_policy
    }

    fn load(&self, model: &InferenceModel<Self>) -> Result<TextualState, MlError> {
        textual_state(model)
    }

    fn session(loaded: &TextualState) -> &OrtSession {
        &loaded.session
    }

    fn predict(
        &self,
        _: &InferenceModel<Self>,
        state: &TextualState,
        inputs: &[Input],
        options: &Options,
    ) -> Result<Output, MlError> {
        let tokens = Self::tokenize(state, inputs[0].text()?, language(options))?;
        run_textual(state, &self.shape_policy, tokens)
    }
}

pub struct MClipTextualEncoder {
    shape_policy: ShapePolicy,
}

impl MClipTextualEncoder {
    pub fn new() -> Self {
        Self {
            shape_policy: ShapePolicy::default(),
        }
    }

    pub fn tokenize(state: &TextualState, text: &str) -> Result<HashMap<String, Tensor>, MlError> {
        let text = clean_text(text, state.canonicalize);
        let tokens = encode(&state.tokenizer, &text)?;
        Ok(HashMap::from([
            ("input_ids".to_owned(), row(tokens.get_ids())),
            (
                "attention_mask".to_owned(),
                row(tokens.get_attention_mask()),
            ),
        ]))
    }
}

impl Default for MClipTextualEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Kind for MClipTextualEncoder {
    type Loaded = TextualState;
    const NAME: &'static str = "MClipTextualEncoder";
    const IDENTITY: ModelIdentity = (ModelType::Textual, ModelTask::Search);
    const SOURCES: &'static [ModelSource] = &[ModelSource::Mclip];
    const THREADS: usize = 4;

    fn shape_policy(&self) -> &ShapePolicy {
        &self.shape_policy
    }

    fn load(&self, model: &InferenceModel<Self>) -> Result<TextualState, MlError> {
        textual_state(model)
    }

    fn session(loaded: &TextualState) -> &OrtSession {
        &loaded.session
    }

    fn predict(
        &self,
        _: &InferenceModel<Self>,
        state: &TextualState,
        inputs: &[Input],
        _: &Options,
    ) -> Result<Output, MlError> {
        let tokens = Self::tokenize(state, inputs[0].text()?)?;
        run_textual(state, &self.shape_policy, tokens)
    }
}
