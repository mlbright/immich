use std::path::Path;

use anyhow::Context;
use sha2::{Digest, Sha256};
use tracing::{error, info, warn};

use crate::config::settings;
use crate::huggingface_hub::{client, download_file};
use crate::models::base::{InferenceModel, Kind, MlError};
use crate::schemas::ModelFormat;

const RAPIDOCR_MODELS: &[(&str, &str, &str, &str)] = &[
    (
        "CH__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "CH__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    ),
    (
        "CH__PP-OCRv5_server",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
        "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
    ),
    (
        "CH__PP-OCRv5_server",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    ),
    (
        "EL__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "EL__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/el_PP-OCRv5_rec_mobile.onnx",
        "b4368bccd557123c702b7549fee6cd1e94b581337d1c9b65310f109131542b7f",
    ),
    (
        "EN__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "EN__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    ),
    (
        "ESLAV__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "ESLAV__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx",
        "08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab",
    ),
    (
        "KOREAN__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "KOREAN__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
        "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b",
    ),
    (
        "LATIN__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "LATIN__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
        "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a",
    ),
    (
        "PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    ),
    (
        "PP-OCRv5_server",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
        "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
    ),
    (
        "PP-OCRv5_server",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    ),
    (
        "PP-OCRv6_medium",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx",
        "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2",
    ),
    (
        "PP-OCRv6_medium",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
        "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
    ),
    (
        "PP-OCRv6_small",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    (
        "PP-OCRv6_small",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
    (
        "PP-OCRv6_tiny",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
        "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
    ),
    (
        "PP-OCRv6_tiny",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
        "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b",
    ),
    (
        "TH__PP-OCRv5_mobile",
        "detection",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    (
        "TH__PP-OCRv5_mobile",
        "recognition",
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/th_PP-OCRv5_rec_mobile.onnx",
        "de541dd83161c241ff426f7ecfd602a0ba77d686cf3ab9a6c255ea82fd08006e",
    ),
];

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DownloadError(pub String);

pub fn model_format_default() -> ModelFormat {
    ModelFormat::Onnx
}

fn file_sha256(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn download<K: Kind>(model: &InferenceModel<K>) -> Result<(), MlError> {
    if !settings().legacy_models() || model.cached() {
        return model.download_snapshot();
    }
    let model_type = model.model_type().value();
    let (_, _, url, sha256) = RAPIDOCR_MODELS
        .iter()
        .find(|(name, kind, _, _)| *name == model.model_name && *kind == model_type)
        .with_context(|| format!("Unsupported configuration for '{}'", model.model_name))?;
    let save_path = model.model_path();
    info!(
        "Downloading {model_type} model '{}' to {}. This may take a while.",
        model.model_name,
        save_path.display()
    );
    if let Some(parent) = save_path.parent() {
        std::fs::create_dir_all(parent).map_err(anyhow::Error::from)?;
    }
    if save_path.exists() {
        if file_sha256(&save_path).map_err(anyhow::Error::from)? == *sha256 {
            info!("File exists and is valid: {}", save_path.display());
            return Ok(());
        }
        warn!(
            "File exists but is invalid, redownloading: {}",
            save_path.display()
        );
    }
    info!("Initiating download: {url}");
    tokio::runtime::Handle::current()
        .block_on(async { download_file(&client()?, url, &save_path, false).await })
        .map_err(|cause| {
            error!("Download failed: {url}");
            anyhow::Error::new(DownloadError(format!("Failed to download {url}")))
                .context(cause.to_string())
        })?;
    info!("Successfully saved to: {}", save_path.display());
    Ok(())
}
