use std::convert::Infallible;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::handler::Handler;
use axum::http::{HeaderValue, Method, Request, StatusCode, Uri, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{MethodRouter, get, post};
use tower::ServiceExt;

use crate::starlette;
use crate::uvicorn::Scope;

const OPENAPI: &str = include_str!("fastapi/openapi.json");
const SWAGGER_UI: &str = include_str!("fastapi/swagger_ui.html");
const SWAGGER_UI_OAUTH2_REDIRECT: &str = include_str!("fastapi/swagger_ui_oauth2_redirect.html");
const REDOC: &str = include_str!("fastapi/redoc.html");

pub fn json_response(status: StatusCode, body: impl IntoResponse) -> Response {
    let mut response = (status, body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

pub fn detail(status: u16, detail: &str) -> Response {
    let body = serde_json::json!({ "detail": detail }).to_string();
    json_response(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        body,
    )
}

fn with_path(uri: &Uri, path: &str) -> Option<Uri> {
    match uri.query() {
        Some(query) => format!("{path}?{query}").parse().ok(),
        None => path.parse().ok(),
    }
}

fn separate_allowed_methods(response: &mut Response) {
    let separated = response
        .headers()
        .get(header::ALLOW)
        .and_then(|allow| allow.to_str().ok())
        .and_then(|allow| HeaderValue::from_str(&allow.replace(',', ", ")).ok());
    if let Some(allow) = separated {
        response.headers_mut().insert(header::ALLOW, allow);
    }
}

#[derive(Clone)]
struct Route {
    path: &'static str,
    method: Option<Method>,
}

#[derive(Clone)]
pub struct FastApi<S = ()> {
    router: Router<S>,
    routes: Arc<Vec<Route>>,
}

impl<S: Clone + Send + Sync + 'static> Default for FastApi<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone + Send + Sync + 'static> FastApi<S> {
    pub fn new() -> Self {
        Self {
            router: Router::new(),
            routes: Arc::default(),
        }
        .route(
            "/openapi.json",
            None,
            get(|| async { json_response(StatusCode::OK, OPENAPI) }),
        )
        .route("/docs", None, get(|| async { Html(SWAGGER_UI) }))
        .route(
            "/docs/oauth2-redirect",
            None,
            get(|| async { Html(SWAGGER_UI_OAUTH2_REDIRECT) }),
        )
        .route("/redoc", None, get(|| async { Html(REDOC) }))
    }

    fn route(
        mut self,
        path: &'static str,
        method: Option<Method>,
        method_router: MethodRouter<S>,
    ) -> Self {
        self.router = self.router.route(path, method_router);
        Arc::make_mut(&mut self.routes).push(Route { path, method });
        self
    }

    pub fn get<H: Handler<T, S>, T: 'static>(self, path: &'static str, handler: H) -> Self {
        self.route(path, Some(Method::GET), get(handler))
    }

    pub fn post<H: Handler<T, S>, T: 'static>(self, path: &'static str, handler: H) -> Self {
        self.route(path, Some(Method::POST), post(handler))
    }

    pub fn with_state(self, state: S) -> FastApi {
        FastApi {
            router: self
                .router
                .method_not_allowed_fallback(|| async { detail(405, "Method Not Allowed") })
                .layer(DefaultBodyLimit::disable())
                .with_state(state),
            routes: self.routes,
        }
    }
}

fn method_not_allowed(allowed: &Method) -> Response {
    let mut response = detail(405, "Method Not Allowed");
    if let Ok(allow) = HeaderValue::from_str(allowed.as_str()) {
        response.headers_mut().insert(header::ALLOW, allow);
    }
    response
}

impl FastApi {
    fn find(&self, path: &str) -> Option<&Route> {
        self.routes.iter().find(|route| route.path == path)
    }

    pub async fn call(self, mut request: Request<Body>, scope: Scope) -> Response {
        let Some(route) = self.find(&scope.path) else {
            return starlette::redirect_slashes(&scope, &request, |path| self.find(path).is_some())
                .unwrap_or_else(|| detail(404, "Not Found"));
        };
        if let Some(method) = &route.method
            && request.method() != method
        {
            return method_not_allowed(method);
        }
        if request.uri().path() != scope.path
            && let Some(uri) = with_path(request.uri(), &scope.path)
        {
            *request.uri_mut() = uri;
        }
        let Ok::<_, Infallible>(mut response) = self.router.oneshot(request).await;
        separate_allowed_methods(&mut response);
        response
    }
}
