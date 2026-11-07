use std::net::{IpAddr, TcpListener};
use std::os::fd::FromRawFd;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::Context;
use immich_ml::config::{NON_PREFIXED_SETTINGS, log_level, settings};
use immich_ml::gunicorn::{self, Config, HEARTBEAT, LISTENER_FD, WORKER_BOOT_ERROR, WorkerTmp};
use immich_ml::main::{WorkerExit, work};
use immich_ml::sessions::ort::flush_denormals;
use immich_ml::sessions::prepare;
use tokio::signal::unix::{SignalKind, signal};
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

    match std::env::args().nth(1).as_deref() {
        Some("prepare") => return Ok(prepare::main()),
        Some("worker") => return worker(),
        _ => {}
    }

    match std::env::var("IMMICH_SOURCE_REF") {
        Ok(source_ref) if !source_ref.is_empty() => info!("Initializing Immich ML [{source_ref}]"),
        _ => info!("Initializing Immich ML"),
    }
    let settings = settings();
    let host = bind_host(&non_prefixed.immich_host);
    let listener =
        TcpListener::bind((host.as_str(), non_prefixed.immich_port)).with_context(|| {
            format!(
                "cannot bind {}:{}",
                non_prefixed.immich_host, non_prefixed.immich_port
            )
        })?;
    info!(
        "Listening at: http://{} ({})",
        listener.local_addr()?,
        std::process::id()
    );
    let device_ids = std::env::var("MACHINE_LEARNING_DEVICE_IDS")
        .unwrap_or_else(|_| "0".to_owned())
        .replace(' ', "")
        .split(',')
        .map(str::to_owned)
        .collect();
    gunicorn::run(
        listener,
        Config {
            workers: settings.workers.max(0) as usize,
            timeout: Duration::from_secs(settings.worker_timeout.max(0) as u64),
            graceful_timeout: Duration::from_secs(10),
            device_ids,
        },
    )
}

fn worker() -> anyhow::Result<ExitCode> {
    let fd: i32 = std::env::var(LISTENER_FD)
        .context("a worker needs the arbiter's listening socket")?
        .parse()?;
    let listener = unsafe { TcpListener::from_raw_fd(fd) };
    unsafe {
        libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    listener.set_nonblocking(true)?;
    let tmp = std::env::var_os(HEARTBEAT).map(|path| WorkerTmp::open(Path::new(&path)));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .on_thread_start(flush_denormals)
        .build()?;
    let exit = runtime.block_on(async {
        flush_denormals();
        let mut quit = signal(SignalKind::quit())?;
        tokio::spawn(async move {
            quit.recv().await;
            std::process::exit(0);
        });
        work(tokio::net::TcpListener::from_std(listener)?, tmp).await
    })?;
    Ok(match exit {
        WorkerExit::Finished => ExitCode::SUCCESS,
        WorkerExit::BootError => ExitCode::from(WORKER_BOOT_ERROR),
    })
}
