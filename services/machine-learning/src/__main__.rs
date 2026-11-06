use std::net::IpAddr;
use std::process::ExitCode;

use anyhow::Context;
use immich_ml::config::{NON_PREFIXED_SETTINGS, log_level, settings};
use immich_ml::sessions::ort::flush_denormals;
use immich_ml::sessions::prepare;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn bind_host(host: &str) -> String {
    let unbracketed = host.trim_start_matches('[').trim_end_matches(']');
    match unbracketed.parse::<IpAddr>() {
        Ok(address) => address.to_string(),
        Err(_) => host.to_owned(),
    }
}

fn main() -> anyhow::Result<ExitCode> {
    let non_prefixed = &*NON_PREFIXED_SETTINGS;
    let level = log_level(&non_prefixed.immich_log_level);
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::level_filters::LevelFilter::WARN.into())
        .parse_lossy(format!("immich_ml={level},immich_ml_bin={level},ort=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(!non_prefixed.no_color)
        .with_target(false)
        .init();

    if std::env::args().nth(1).as_deref() == Some("prepare") {
        return Ok(prepare::main());
    }

    match std::env::var("IMMICH_SOURCE_REF") {
        Ok(source_ref) if !source_ref.is_empty() => info!("Initializing Immich ML [{source_ref}]"),
        _ => info!("Initializing Immich ML"),
    }
    let _ = settings();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .on_thread_start(flush_denormals)
        .build()?;
    runtime.block_on(async {
        flush_denormals();
        let host = bind_host(&non_prefixed.immich_host);
        let listener = tokio::net::TcpListener::bind((host.as_str(), non_prefixed.immich_port))
            .await
            .with_context(|| {
                format!(
                    "cannot bind {}:{}",
                    non_prefixed.immich_host, non_prefixed.immich_port
                )
            })?;
        info!("Listening at: http://{}", listener.local_addr()?);
        immich_ml::main::serve(listener).await
    })?;
    Ok(ExitCode::SUCCESS)
}
