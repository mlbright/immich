use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::FutureExt;
use futures::future::{BoxFuture, Shared, join_all};
use tokio::sync::Semaphore;
use tokio::task::AbortHandle;
use tracing::{error, info, warn};

use crate::allocator;
use crate::config::{PreloadModelData, settings};
use crate::models::base::{AnyModel, InferenceEntry, Input, MlError};
use crate::models::cache::ModelCache;
use crate::models::transforms::decode_pil;
use crate::pipeline::{Clip, FacialRecognition, Ocr, PipelineRequest, Slot};
use crate::schemas::{
    FaceDetectionOptions, FaceRecognitionOptions, InferenceResponse, ModelFormat, ModelIdentity,
    Output, TextDetectionOptions, TextRecognitionOptions, TextualOptions, VisualOptions,
    identity_repr,
};

#[derive(Default)]
struct RequestState {
    last_called: Option<Instant>,
    release: Option<AbortHandle>,
}

pub struct App {
    pub model_cache: ModelCache,
    thread_pool: Option<Arc<Semaphore>>,
    active_requests: AtomicUsize,
    state: Mutex<RequestState>,
}

type Run = Shared<BoxFuture<'static, Result<Arc<Output>, Arc<MlError>>>>;

impl App {
    pub fn new() -> Arc<Self> {
        let settings = settings();
        let thread_pool = (settings.request_threads > 0)
            .then(|| Arc::new(Semaphore::new(settings.request_threads as usize)));
        Arc::new(Self {
            model_cache: ModelCache::new(),
            thread_pool,
            active_requests: AtomicUsize::new(0),
            state: Mutex::new(RequestState::default()),
        })
    }

    async fn run<R: Send + 'static>(&self, func: impl FnOnce() -> R + Send + 'static) -> R {
        let permit = match &self.thread_pool {
            Some(pool) => Some(
                pool.clone()
                    .acquire_owned()
                    .await
                    .expect("thread pool semaphore"),
            ),
            None => None,
        };
        let result = tokio::task::spawn_blocking(move || {
            let result = func();
            drop(permit);
            result
        })
        .await;
        match result {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic.into_panic()),
        }
    }

    fn enter(self: &Arc<Self>) -> ActiveRequest {
        self.active_requests.fetch_add(1, Ordering::SeqCst);
        let mut state = self.state.lock().expect("request state lock");
        state.last_called = Some(Instant::now());
        if let Some(release) = state.release.take() {
            release.abort();
        }
        ActiveRequest { app: self.clone() }
    }
}

struct ActiveRequest {
    app: Arc<App>,
}

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        if self.app.active_requests.fetch_sub(1, Ordering::SeqCst) == 1 {
            let mut state = self.app.state.lock().expect("request state lock");
            state.release = Some(
                tokio::spawn(async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    allocator::release();
                })
                .abort_handle(),
            );
        }
    }
}

pub async fn lifespan(app: &Arc<App>) -> anyhow::Result<()> {
    let settings = settings();
    info!(
        "Created in-memory cache with unloading {}.",
        if settings.model_ttl > 0 {
            format!("after {}s of inactivity", settings.model_ttl)
        } else {
            "disabled".to_owned()
        }
    );
    if settings.request_threads > 0 {
        info!(
            "Initialized request thread pool with {} threads.",
            settings.request_threads
        );
    }
    if settings.model_ttl > 0 && settings.model_ttl_poll_s > 0 {
        tokio::spawn(idle_shutdown_task(app.clone()));
    }
    if let Some(preload) = &settings.preload {
        preload_models(app, preload).await?;
        allocator::release();
    }
    Ok(())
}

fn names(setting: &Option<String>) -> Vec<String> {
    setting
        .as_deref()
        .map(|setting| {
            setting
                .split(',')
                .map(|name| name.trim().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

async fn preload_models(app: &Arc<App>, preload: &PreloadModelData) -> anyhow::Result<()> {
    let mut requests = Vec::new();
    for name in names(&preload.clip.textual) {
        requests.push(PipelineRequest {
            clip: Clip {
                textual: Some(Slot::new(name, TextualOptions::default())),
                ..Clip::default()
            },
            ..PipelineRequest::default()
        });
    }
    for name in names(&preload.clip.visual) {
        requests.push(PipelineRequest {
            clip: Clip {
                visual: Some(Slot::new(name, VisualOptions::default())),
                ..Clip::default()
            },
            ..PipelineRequest::default()
        });
    }
    for name in names(&preload.facial_recognition.detection) {
        requests.push(PipelineRequest {
            facial_recognition: FacialRecognition {
                detection: Some(Slot::new(name, FaceDetectionOptions::default())),
                ..FacialRecognition::default()
            },
            ..PipelineRequest::default()
        });
    }
    for name in names(&preload.facial_recognition.recognition) {
        requests.push(PipelineRequest {
            facial_recognition: FacialRecognition {
                recognition: Some(Slot::new(name, FaceRecognitionOptions::default())),
                ..FacialRecognition::default()
            },
            ..PipelineRequest::default()
        });
    }
    for name in names(&preload.ocr.detection) {
        let options = TextDetectionOptions {
            max_resolution: preload.ocr.max_resolution,
            ..TextDetectionOptions::default()
        };
        requests.push(PipelineRequest {
            ocr: Ocr {
                detection: Some(Slot::new(name, options)),
                ..Ocr::default()
            },
            ..PipelineRequest::default()
        });
    }
    for name in names(&preload.ocr.recognition) {
        requests.push(PipelineRequest {
            ocr: Ocr {
                recognition: Some(Slot::new(name, TextRecognitionOptions::default())),
                ..Ocr::default()
            },
            ..PipelineRequest::default()
        });
    }
    for request in requests {
        for entry in request.entries()? {
            info!("Preloading: {entry}");
            let model = load(app, app.model_cache.get(&entry, None))
                .await
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            let built = model.clone();
            attempt(app, &model, move || built.build(), false)
                .await
                .map_err(|error| anyhow::anyhow!("{error}"))?;
        }
    }
    Ok(())
}

fn get_entries(entries: &str) -> Result<Vec<InferenceEntry>, MlError> {
    let found = serde_json::from_str::<PipelineRequest>(entries)
        .map_err(anyhow::Error::from)
        .and_then(|request| request.entries())
        .map_err(|error| {
            error!("Invalid request format: {error}");
            MlError::http(422, "Invalid request format.")
        })?;
    if found.is_empty() {
        return Err(MlError::http(422, "No model task requested."));
    }
    Ok(found)
}

fn detail(status: u16, detail: &str) -> Response {
    let body = serde_json::json!({ "detail": detail }).to_string();
    json_response(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        body,
    )
}

fn json_response(status: StatusCode, body: String) -> Response {
    let mut response = (status, body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

fn error_response(error: &MlError) -> Response {
    match error {
        MlError::Http {
            status,
            detail: message,
        } => detail(*status, message),
        other => {
            error!("Exception in ASGI application: {other:#}");
            let mut response =
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response();
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            response
        }
    }
}

fn missing_entries() -> Response {
    let body = r#"{"detail":[{"type":"missing","loc":["body","entries"],"msg":"Field required","input":null}]}"#;
    json_response(StatusCode::UNPROCESSABLE_ENTITY, body.to_owned())
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/ping", get(ping))
        .route("/predict", post(predict))
        .method_not_allowed_fallback(|| async { detail(405, "Method Not Allowed") })
        .fallback(|| async { detail(404, "Not Found") })
        .layer(DefaultBodyLimit::disable())
        .with_state(app)
}

async fn root() -> Response {
    json_response(StatusCode::OK, r#"{"message":"Immich ML"}"#.to_owned())
}

async fn ping() -> Response {
    let mut response = "pong".into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}

async fn predict(
    State(app): State<Arc<App>>,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Response {
    let _active = app.enter();
    let mut entries = None;
    let mut image = None;
    let mut text = None;
    if let Ok(mut multipart) = multipart {
        loop {
            match multipart.next_field().await {
                Ok(Some(field)) => {
                    let is_file = field.file_name().is_some();
                    match field.name().unwrap_or_default().to_owned().as_str() {
                        "entries" => {
                            entries = field.text().await.ok().filter(|value| !value.is_empty())
                        }
                        "image" => {
                            image = field
                                .bytes()
                                .await
                                .ok()
                                .filter(|value| is_file || !value.is_empty())
                        }
                        "text" => text = field.text().await.ok().filter(|value| !value.is_empty()),
                        _ => {}
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    warn!("Failed to parse form: {error}");
                    break;
                }
            }
        }
    }
    let Some(entries) = entries else {
        return missing_entries();
    };
    let entries = match get_entries(&entries) {
        Ok(entries) => entries,
        Err(error) => return error_response(&error),
    };
    let payload = if let Some(image) = image {
        let decoded = app.run(move || decode_pil(&image)).await;
        match decoded {
            Ok(decoded) if decoded.width() == 0 || decoded.height() == 0 => {
                return detail(400, "Image has zero width or height");
            }
            Ok(decoded) => Input::Image(Arc::new(decoded)),
            Err(error) => return error_response(&MlError::Other(error)),
        }
    } else if let Some(text) = text {
        Input::Text(text.into())
    } else {
        return detail(400, "Either image or text must be provided");
    };
    match run_inference(&app, payload, entries).await {
        Ok(response) => json_response(
            StatusCode::OK,
            serde_json::to_string(&response).expect("response serializes"),
        ),
        Err(error) => error_response(&error),
    }
}

pub async fn run_inference(
    app: &Arc<App>,
    payload: Input,
    entries: Vec<InferenceEntry>,
) -> Result<InferenceResponse, MlError> {
    let settings = settings();
    let identities: HashSet<ModelIdentity> =
        entries.iter().map(|entry| entry.model.identity()).collect();
    let mut read: HashSet<ModelIdentity> = HashSet::new();
    let mut runs: HashMap<ModelIdentity, Run> = HashMap::new();
    let mut tasks: Vec<Run> = Vec::new();
    for entry in &entries {
        let model = app.model_cache.get(entry, Some(settings.model_ttl));
        read.extend(model.depends().iter().copied());
        let missing = model
            .depends()
            .iter()
            .find(|dependency| !identities.contains(dependency))
            .copied();
        let dependencies: Vec<Run> = model
            .depends()
            .iter()
            .filter_map(|dependency| runs.get(dependency).cloned())
            .collect();
        let (app, payload, options, name) = (
            app.clone(),
            payload.clone(),
            entry.options.clone(),
            entry.model.name(),
        );
        let task = async move {
            if let Some(missing) = missing {
                return Err(Arc::new(MlError::http(
                    400,
                    format!("{name} depends on output of {}", identity_repr(missing)),
                )));
            }
            let mut inputs = vec![payload];
            for dependency in dependencies {
                inputs.push(Input::Output(dependency.await?));
            }
            let model = load(&app, model).await.map_err(Arc::new)?;
            let predicting = model.clone();
            let output = attempt(
                &app,
                &model,
                move || predicting.predict(&inputs, &options),
                true,
            )
            .await
            .map_err(Arc::new)?;
            Ok(Arc::new(output))
        }
        .boxed()
        .shared();
        runs.insert(entry.model.identity(), task.clone());
        tasks.push(task);
    }
    let outputs = join_all(tasks).await;
    let mut collected = Vec::with_capacity(outputs.len());
    for output in outputs {
        collected.push(output.map_err(|error| match Arc::try_unwrap(error) {
            Ok(error) => error,
            Err(error) => match error.as_ref() {
                MlError::Http { status, detail } => MlError::http(*status, detail.clone()),
                other => MlError::Other(anyhow::anyhow!("{other:#}")),
            },
        })?);
    }
    let mut response = InferenceResponse {
        entries: Vec::new(),
        image_size: None,
    };
    for (entry, output) in entries.iter().zip(collected) {
        if read.contains(&entry.model.identity()) {
            continue;
        }
        let task = entry.model.identity().1;
        match response
            .entries
            .iter_mut()
            .find(|(existing, _)| *existing == task)
        {
            Some(slot) => slot.1 = output,
            None => response.entries.push((task, output)),
        }
    }
    if let Input::Image(image) = &payload {
        response.image_size = Some((image.height(), image.width()));
    }
    Ok(response)
}

async fn load(app: &Arc<App>, model: Arc<dyn AnyModel>) -> Result<Arc<dyn AnyModel>, MlError> {
    if !model.loaded() {
        let loading = model.clone();
        attempt(app, &model, move || loading.load(), false).await?;
    }
    Ok(model)
}

async fn attempt<R: Send + 'static>(
    app: &Arc<App>,
    model: &Arc<dyn AnyModel>,
    func: impl Fn() -> Result<R, MlError> + Send + Sync + 'static,
    predicting: bool,
) -> Result<R, MlError> {
    let func = Arc::new(func);
    let try_once = {
        let model = model.clone();
        let func = func.clone();
        move || -> Result<R, MlError> {
            if !model.loaded() && model.load_attempts() > 1 {
                return Err(MlError::http(
                    500,
                    format!("Failed to load model '{}'", model.model_name()),
                ));
            }
            match func() {
                Err(MlError::FileNotFound(message)) => {
                    if model.model_format() == ModelFormat::Onnx {
                        return Err(MlError::FileNotFound(message));
                    }
                    warn!(
                        "{} is available, but model '{}' does not support it.",
                        model.model_format().value().to_uppercase(),
                        model.model_name()
                    );
                    model.unload();
                    model.set_model_format(ModelFormat::Onnx);
                    func()
                }
                other => other,
            }
        }
    };
    let try_once = Arc::new(try_once);
    let first = try_once.clone();
    match app.run(move || first()).await {
        Err(error) if is_corrupt(&error, predicting) => {
            warn!(
                "Failed to load {} model '{}', clearing cache: {error}",
                model.model_type().value().replace('_', " "),
                model.model_name()
            );
            model.unload();
            let clearing = model.clone();
            app.run(move || clearing.clear_cache()).await?;
            let second = try_once.clone();
            app.run(move || second()).await
        }
        other => other,
    }
}

fn is_corrupt(error: &MlError, predicting: bool) -> bool {
    error.is_model_file_error() || (!predicting && error.is_os_error())
}

async fn idle_shutdown_task(app: Arc<App>) {
    let settings = settings();
    loop {
        let idle = {
            let state = app.state.lock().expect("request state lock");
            state.last_called.is_some_and(|last_called| {
                app.active_requests.load(Ordering::SeqCst) == 0
                    && last_called.elapsed().as_secs_f64() > settings.model_ttl as f64
            })
        };
        if idle {
            info!("Unloading models due to inactivity.");
            app.model_cache.clear_expiring();
            app.state.lock().expect("request state lock").last_called = None;
        }
        tokio::time::sleep(Duration::from_secs(settings.model_ttl_poll_s as u64)).await;
    }
}

pub async fn serve(listener: tokio::net::TcpListener) -> anyhow::Result<()> {
    let app = App::new();
    lifespan(&app).await?;
    let cache = app.model_cache.clone();
    axum::serve(listener, router(app))
        .with_graceful_shutdown(async {
            let interrupt = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("signal handler");
                tokio::select! {
                    _ = interrupt => {}
                    _ = terminate.recv() => {}
                }
            }
            #[cfg(not(unix))]
            {
                let _ = interrupt.await;
            }
        })
        .await?;
    cache.clear();
    Ok(())
}
