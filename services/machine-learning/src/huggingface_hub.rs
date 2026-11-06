use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

#[derive(Deserialize)]
struct RepoInfo {
    sha: Option<String>,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_owned())
        .trim_end_matches('/')
        .to_owned()
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|token| !token.is_empty())
}

fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .is_ok_and(|value| matches!(value.to_lowercase().as_str(), "1" | "on" | "yes" | "true"))
}

fn ignored(file: &str, ignore_patterns: &[&str]) -> bool {
    ignore_patterns
        .iter()
        .any(|pattern| match pattern.strip_prefix('*') {
            Some(suffix) => file.ends_with(suffix),
            None => file == *pattern,
        })
}

pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(concat!("immich-ml/", env!("CARGO_PKG_VERSION")))
        .build()?)
}

pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    destination: &Path,
    authorized: bool,
) -> Result<()> {
    let mut request = client.get(url);
    if authorized && let Some(token) = token() {
        request = request.bearer_auth(token);
    }
    let mut response = request.send().await?.error_for_status()?;
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let partial = destination.with_extension(format!(
        "{}incomplete",
        destination
            .extension()
            .map(|extension| format!("{}.", extension.to_string_lossy()))
            .unwrap_or_default()
    ));
    let mut file = tokio::fs::File::create(&partial).await?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&partial, destination).await?;
    Ok(())
}

pub fn snapshot_download(
    repo_id: &str,
    revision: &str,
    local_dir: &Path,
    ignore_patterns: &[&str],
) -> Result<()> {
    if offline() {
        bail!("Cannot find an appropriate cached snapshot folder for '{repo_id}' while offline");
    }
    tokio::runtime::Handle::current().block_on(async {
        let client = client()?;
        let endpoint = endpoint();
        let mut request = client.get(format!(
            "{endpoint}/api/models/{repo_id}/revision/{revision}"
        ));
        if let Some(token) = token() {
            request = request.bearer_auth(token);
        }
        let info: RepoInfo = request
            .send()
            .await?
            .error_for_status()
            .with_context(|| {
                format!("Repository Not Found for '{repo_id}' at revision '{revision}'")
            })?
            .json()
            .await?;
        let commit = info.sha.unwrap_or_else(|| revision.to_owned());
        if revision != commit {
            let reference = local_dir
                .join(format!("models--{}", repo_id.replace('/', "--")))
                .join("refs")
                .join(revision);
            if let Some(parent) = reference.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&reference, &commit).await?;
        }
        for sibling in info
            .siblings
            .iter()
            .filter(|sibling| !ignored(&sibling.rfilename, ignore_patterns))
        {
            let url = format!(
                "{endpoint}/{repo_id}/resolve/{commit}/{}",
                sibling.rfilename
            );
            download_file(&client, &url, &local_dir.join(&sibling.rfilename), true)
                .await
                .with_context(|| format!("failed to download {}", sibling.rfilename))?;
        }
        Ok(())
    })
}
