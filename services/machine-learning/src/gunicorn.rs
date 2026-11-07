use std::fs::{self, File};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::signal::unix::{SignalKind, signal};
use tracing::{error, info};

pub const WORKER_BOOT_ERROR: u8 = 3;
pub const LISTENER_FD: &str = "IMMICH_ML_LISTENER_FD";
pub const HEARTBEAT: &str = "IMMICH_ML_HEARTBEAT";

pub fn forwarded_allow_ips() -> String {
    std::env::var("FORWARDED_ALLOW_IPS").unwrap_or_else(|_| "127.0.0.1,::1".to_owned())
}

pub struct Config {
    pub workers: usize,
    pub timeout: Duration,
    pub graceful_timeout: Duration,
    pub device_ids: Vec<String>,
}

pub struct WorkerTmp {
    path: PathBuf,
}

impl WorkerTmp {
    fn create(index: u64) -> Result<Self> {
        let path =
            std::env::temp_dir().join(format!("immich-ml-worker-{}-{index}", std::process::id()));
        File::create(&path)?;
        Ok(Self { path })
    }

    pub fn open(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    pub fn notify(&self) {
        if let Ok(file) = File::options().write(true).open(&self.path) {
            let _ = file.set_modified(SystemTime::now());
        }
    }

    fn last_update(&self) -> Option<SystemTime> {
        fs::metadata(&self.path)
            .and_then(|meta| meta.modified())
            .ok()
    }
}

impl Drop for WorkerTmp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct Worker {
    child: Child,
    pid: i32,
    tmp: WorkerTmp,
    aborted: bool,
}

fn kill(pid: i32, signal: libc::c_int) {
    unsafe {
        libc::kill(pid, signal);
    }
}

fn inheritable(fd: RawFd) -> Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0 {
        return Err(std::io::Error::last_os_error()).context("cannot share the listening socket");
    }
    Ok(())
}

fn signal_name(signal: i32) -> String {
    match signal {
        libc::SIGKILL => "SIGKILL".to_owned(),
        libc::SIGTERM => "SIGTERM".to_owned(),
        libc::SIGABRT => "SIGABRT".to_owned(),
        libc::SIGSEGV => "SIGSEGV".to_owned(),
        libc::SIGINT => "SIGINT".to_owned(),
        libc::SIGQUIT => "SIGQUIT".to_owned(),
        other => format!("signal {other}"),
    }
}

struct Arbiter {
    config: Config,
    fd: RawFd,
    workers: Vec<Worker>,
    spawned: u64,
}

impl Arbiter {
    fn spawn_worker(&mut self) -> Result<()> {
        let tmp = WorkerTmp::create(self.spawned)?;
        self.spawned += 1;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("worker")
            .env(LISTENER_FD, self.fd.to_string())
            .env(HEARTBEAT, &tmp.path);
        if !self.config.device_ids.is_empty() {
            let index = self.workers.len() % self.config.device_ids.len();
            command.env("MACHINE_LEARNING_DEVICE_ID", &self.config.device_ids[index]);
        }
        let child = command.spawn()?;
        let pid = child.id().context("worker exited at once")? as i32;
        info!("Booting worker with pid: {pid}");
        self.workers.push(Worker {
            child,
            pid,
            tmp,
            aborted: false,
        });
        Ok(())
    }

    fn manage_workers(&mut self) -> Result<()> {
        while self.workers.len() < self.config.workers {
            self.spawn_worker()?;
        }
        Ok(())
    }

    fn reap_workers(&mut self) -> Option<u8> {
        let mut halt = None;
        self.workers.retain_mut(|worker| {
            let Ok(Some(status)) = worker.child.try_wait() else {
                return true;
            };
            if let Some(code) = status.code() {
                if code != 0 {
                    error!("Worker (pid:{}) exited with code {code}.", worker.pid);
                }
                if code == i32::from(WORKER_BOOT_ERROR) {
                    halt = Some(WORKER_BOOT_ERROR);
                }
            } else if let Some(signal) = status.signal() {
                if signal == libc::SIGKILL {
                    error!(
                        "Worker (pid:{}) was sent SIGKILL! Perhaps out of memory?",
                        worker.pid
                    );
                } else if signal != libc::SIGTERM {
                    error!(
                        "Worker (pid:{}) was sent {}!",
                        worker.pid,
                        signal_name(signal)
                    );
                }
            }
            false
        });
        halt
    }

    fn murder_workers(&mut self) {
        if self.config.timeout.is_zero() {
            return;
        }
        for worker in &mut self.workers {
            let Some(updated) = worker.tmp.last_update() else {
                continue;
            };
            let silent = SystemTime::now()
                .duration_since(updated)
                .unwrap_or_default();
            if silent <= self.config.timeout {
                continue;
            }
            if !worker.aborted {
                error!("WORKER TIMEOUT (pid:{})", worker.pid);
                worker.aborted = true;
                kill(worker.pid, libc::SIGABRT);
            } else {
                kill(worker.pid, libc::SIGKILL);
            }
        }
    }

    async fn stop(&mut self, graceful: bool) {
        let signal = if graceful {
            libc::SIGTERM
        } else {
            libc::SIGQUIT
        };
        for worker in &self.workers {
            kill(worker.pid, signal);
        }
        let limit = Instant::now() + self.config.graceful_timeout;
        while !self.workers.is_empty() && Instant::now() < limit {
            tokio::time::sleep(Duration::from_millis(100)).await;
            self.reap_workers();
        }
        for worker in &mut self.workers {
            kill(worker.pid, libc::SIGKILL);
            let _ = worker.child.wait().await;
        }
        self.workers.clear();
    }

    async fn run(&mut self) -> Result<ExitCode> {
        let mut terminate = signal(SignalKind::terminate())?;
        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        self.manage_workers()?;
        loop {
            tokio::select! {
                _ = terminate.recv() => {
                    info!("Handling signal: term");
                    self.stop(true).await;
                    break;
                }
                _ = interrupt.recv() => {
                    info!("Handling signal: int");
                    self.stop(false).await;
                    break;
                }
                _ = tick.tick() => {
                    if let Some(code) = self.reap_workers() {
                        self.stop(true).await;
                        error!("Shutting down: Master");
                        error!("Reason: Worker failed to boot.");
                        return Ok(ExitCode::from(code));
                    }
                    self.murder_workers();
                    self.manage_workers()?;
                }
            }
        }
        info!("Shutting down: Master");
        Ok(ExitCode::SUCCESS)
    }
}

pub fn run(listener: std::net::TcpListener, config: Config) -> Result<ExitCode> {
    let fd = listener.as_raw_fd();
    inheritable(fd)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut arbiter = Arbiter {
        config,
        fd,
        workers: Vec::new(),
        spawned: 0,
    };
    let code = runtime.block_on(arbiter.run());
    drop(listener);
    code
}
