use axum::http::{HeaderValue, Request, header};
use axum::response::{IntoResponse, Redirect, Response};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};

use crate::uvicorn::{Scope, latin1};

const REDIRECT_SAFE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~')
    .remove(b':')
    .remove(b'/')
    .remove(b'%')
    .remove(b'#')
    .remove(b'?')
    .remove(b'=')
    .remove(b'@')
    .remove(b'[')
    .remove(b']')
    .remove(b'!')
    .remove(b'$')
    .remove(b'&')
    .remove(b'\'')
    .remove(b'(')
    .remove(b')')
    .remove(b'*')
    .remove(b'+')
    .remove(b',')
    .remove(b';');

pub fn redirect_slashes<B>(
    scope: &Scope,
    request: &Request<B>,
    matches: impl Fn(&str) -> bool,
) -> Option<Response> {
    if scope.path == "/" {
        return None;
    }
    let path = if scope.path.ends_with('/') {
        scope.path.trim_end_matches('/').to_owned()
    } else {
        format!("{}/", scope.path)
    };
    matches(&path).then(|| redirect(&url(scope, request, &path)))
}

fn url<B>(scope: &Scope, request: &Request<B>, path: &str) -> String {
    let scheme = &scope.scheme;
    let mut url = match request.headers().get(header::HOST) {
        Some(host) => format!("{scheme}://{}{path}", latin1(host.as_bytes())),
        None => {
            let default_port = match scheme.as_str() {
                "https" | "wss" => 443,
                _ => 80,
            };
            let host = scope.server.ip();
            match scope.server.port() {
                port if port == default_port => format!("{scheme}://{host}{path}"),
                port => format!("{scheme}://{host}:{port}{path}"),
            }
        }
    };
    if let Some(query) = request.uri().query().filter(|query| !query.is_empty()) {
        url.push('?');
        url.push_str(query);
    }
    url
}

fn redirect(url: &str) -> Response {
    let mut response =
        Redirect::temporary(&utf8_percent_encode(url, REDIRECT_SAFE).to_string()).into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(0));
    response
}
