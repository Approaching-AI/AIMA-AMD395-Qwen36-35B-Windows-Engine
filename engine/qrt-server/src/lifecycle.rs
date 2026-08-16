use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::{error, info};

use crate::api::{
    router, QueueConfig, ServerState, DEFAULT_MAX_QUEUE_DEPTH, DEFAULT_QUEUE_TIMEOUT_SECONDS,
};
use crate::backend::{InferenceBackend, NativeBackend, NATIVE_THREAD_STACK_BYTES};
use crate::tokenizer::{QwenTokenizer, TokenCodec};

const SMOOTH_TAIL_MOE_TOKEN_COUNTS: [usize; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];

#[derive(Clone, Debug, PartialEq, Eq)]
struct SmoothTailMoeBinding {
    tokens: usize,
    provider: PathBuf,
    kernel_dir: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SmoothTailMoeLayout {
    root: PathBuf,
    bindings: Vec<SmoothTailMoeBinding>,
}

#[derive(Clone, Debug, Args)]
pub struct ServeOptions {
    /// Model directory containing config.json, tokenizer.json, and safetensors.
    #[arg(long, env = "QRT_MODEL_PATH")]
    pub model: Option<PathBuf>,

    /// Native whole-provider DLL.
    #[arg(long, env = "QRT_QWEN36_WHOLE_PROVIDER_DLL")]
    pub provider: Option<PathBuf>,

    /// q1024 selected-MoE DLL used by non-retained arbitrary prompt lengths.
    #[arg(long, env = "QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_DLL")]
    pub arbitrary_moe_provider: Option<PathBuf>,

    /// Kernel directory belonging to the arbitrary-prompt q1024 MoE DLL.
    #[arg(long, env = "QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_KERNEL_DIR")]
    pub arbitrary_moe_kernel_dir: Option<PathBuf>,

    /// Root containing packaged q32..q4096 smooth-tail selected-MoE providers.
    #[arg(long, env = "QRT_QWEN36_SMOOTH_TAIL_MOE_ROOT")]
    pub smooth_tail_moe_root: Option<PathBuf>,

    /// Retained runtime KEY=VALUE profile loaded before the model.
    #[arg(long, env = "QRT_RUNTIME_ENV_FILE")]
    pub env_file: Option<PathBuf>,

    /// Runtime KEY=VALUE override applied after --env-file (repeatable).
    #[arg(long = "set-env", value_name = "KEY=VALUE")]
    pub env_overrides: Vec<String>,

    /// Model name exposed through the OpenAI API.
    #[arg(long, default_value = "qwen3.6-35b-a3b", env = "QRT_SERVED_MODEL_NAME")]
    pub model_id: String,

    /// Listen address.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,

    /// Listen port.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Maximum total context length.
    #[arg(long, default_value_t = 262_144)]
    pub max_model_len: usize,

    /// Maximum number of requests waiting behind the active batch-one request.
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_QUEUE_DEPTH,
        env = "QRT_MAX_QUEUE_DEPTH"
    )]
    pub max_queue_depth: usize,

    /// Maximum seconds a request may wait in the inference queue.
    #[arg(
        long,
        default_value_t = DEFAULT_QUEUE_TIMEOUT_SECONDS,
        env = "QRT_QUEUE_TIMEOUT_SECONDS"
    )]
    pub queue_timeout_seconds: u64,

    /// Optional Bearer token required by all model and admin endpoints.
    #[arg(long, env = "QRT_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Permit an unauthenticated non-loopback listener.
    #[arg(long)]
    pub allow_unauthenticated: bool,

    /// Service state file used by start, status, and stop.
    #[arg(long)]
    pub state_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
pub struct StartOptions {
    #[command(flatten)]
    pub serve: ServeOptions,

    /// File receiving detached service stdout and stderr.
    #[arg(long)]
    pub log_file: Option<PathBuf>,

    /// Seconds to wait for model preload and readiness.
    #[arg(long, default_value_t = 60)]
    pub wait_seconds: u64,
}

#[derive(Clone, Debug, Args)]
pub struct StatusOptions {
    #[arg(long)]
    pub state_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
pub struct StopOptions {
    #[arg(long)]
    pub state_file: Option<PathBuf>,

    #[arg(long, env = "QRT_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    #[arg(long, default_value_t = 30)]
    pub wait_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServiceRecord {
    pub schema_version: u32,
    pub status: String,
    pub pid: u32,
    pub address: String,
    pub model: String,
    pub model_path: String,
    pub provider_dll: String,
    #[serde(default)]
    pub arbitrary_moe_provider_dll: Option<String>,
    #[serde(default)]
    pub arbitrary_moe_kernel_dir: Option<String>,
    #[serde(default)]
    pub smooth_tail_moe_root: Option<String>,
    pub max_model_len: usize,
    #[serde(default = "default_record_max_queue_depth")]
    pub max_queue_depth: usize,
    #[serde(default = "default_record_queue_timeout_seconds")]
    pub queue_timeout_seconds: u64,
    pub repo_commit: String,
    pub host: String,
    pub started_unix_seconds: u64,
    pub ready_unix_seconds: Option<u64>,
    pub stopped_unix_seconds: Option<u64>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct HealthIdentity {
    ready: bool,
    pid: u32,
    model: String,
}

pub async fn serve(options: ServeOptions) -> Result<()> {
    if let Some(env_file) = &options.env_file {
        load_env_file(env_file)?;
    }
    apply_env_overrides(&options.env_overrides)?;
    let smooth_tail_moe_root = configure_smooth_tail_moe_providers(
        options.smooth_tail_moe_root.clone(),
        options.env_file.as_deref(),
    )?;
    let model_path = resolve_required_path(
        options.model.clone(),
        "QRT_MODEL_PATH",
        "--model is required (or set QRT_MODEL_PATH)",
    )?;
    let provider_path = resolve_required_path(
        options.provider.clone(),
        "QRT_QWEN36_WHOLE_PROVIDER_DLL",
        "--provider is required (or set QRT_QWEN36_WHOLE_PROVIDER_DLL)",
    )?;
    let (arbitrary_moe_provider, arbitrary_moe_kernel_dir) = configure_arbitrary_moe_provider(
        options.arbitrary_moe_provider.clone(),
        options.arbitrary_moe_kernel_dir.clone(),
    )?;
    if !model_path.is_dir() {
        bail!("model directory does not exist: {}", model_path.display());
    }
    if !provider_path.is_file() {
        bail!("provider DLL does not exist: {}", provider_path.display());
    }
    if options.max_model_len == 0 || options.max_model_len > 262_144 {
        bail!("--max-model-len must be between 1 and 262144");
    }
    if options.max_queue_depth > 4096 {
        bail!("--max-queue-depth must be between 0 and 4096");
    }
    if options.queue_timeout_seconds == 0 || options.queue_timeout_seconds > 86_400 {
        bail!("--queue-timeout-seconds must be between 1 and 86400");
    }
    if !options.host.is_loopback()
        && options.api_key.as_deref().is_none_or(str::is_empty)
        && !options.allow_unauthenticated
    {
        bail!("a non-loopback listener requires --api-key/QRT_API_KEY or --allow-unauthenticated");
    }

    let address = SocketAddr::new(options.host, options.port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| format!("could not bind HTTP listener at {address}"))?;
    let bound_address = listener.local_addr()?;
    let state_file = options
        .state_file
        .clone()
        .unwrap_or_else(default_state_file);
    let started = epoch_seconds();
    let mut record = ServiceRecord {
        schema_version: 1,
        status: "loading".to_owned(),
        pid: std::process::id(),
        address: bound_address.to_string(),
        model: options.model_id.clone(),
        model_path: model_path.display().to_string(),
        provider_dll: provider_path.display().to_string(),
        arbitrary_moe_provider_dll: Some(arbitrary_moe_provider.display().to_string()),
        arbitrary_moe_kernel_dir: Some(arbitrary_moe_kernel_dir.display().to_string()),
        smooth_tail_moe_root: smooth_tail_moe_root.map(|path| path.display().to_string()),
        max_model_len: options.max_model_len,
        max_queue_depth: options.max_queue_depth,
        queue_timeout_seconds: options.queue_timeout_seconds,
        repo_commit: option_env!("QRT_GIT_SHA").unwrap_or("unknown").to_owned(),
        host: machine_name(),
        started_unix_seconds: started,
        ready_unix_seconds: None,
        stopped_unix_seconds: None,
        message: Some("loading tokenizer and native resident model".to_owned()),
    };
    write_state(&state_file, &record)?;
    info!(address = %bound_address, model = %options.model_id, "loading resident model");

    let tokenizer_start = Instant::now();
    let tokenizer: Arc<dyn TokenCodec> = Arc::new(QwenTokenizer::from_model_dir(&model_path)?);
    let tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;
    let model_string = model_path.to_string_lossy().into_owned();
    let provider_string = provider_path.to_string_lossy().into_owned();
    let max_model_len = options.max_model_len;
    let native_loader = std::thread::Builder::new()
        .name("qrt-native-loader".to_owned())
        .stack_size(NATIVE_THREAD_STACK_BYTES)
        .spawn(move || NativeBackend::load(&model_string, &provider_string, max_model_len))
        .context("could not start native model loader thread")?;
    let native = native_loader
        .join()
        .map_err(|_| anyhow!("native model loader thread panicked"))??;
    let load_metrics = native.load_metrics().clone();
    let backend: Arc<dyn InferenceBackend> = Arc::new(native);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let app_state = ServerState::new_with_queue(
        backend,
        tokenizer,
        options.model_id.clone(),
        options.max_model_len,
        options.api_key.filter(|key| !key.is_empty()),
        load_metrics.clone(),
        QueueConfig {
            max_waiting_requests: options.max_queue_depth,
            wait_timeout: Duration::from_secs(options.queue_timeout_seconds),
        },
        shutdown_tx,
    );

    record.status = "ready".to_owned();
    record.ready_unix_seconds = Some(epoch_seconds());
    record.message = Some(format!(
        "resident model ready; tokenizer_ms={tokenizer_ms:.3}; native_load_ms={:.3}",
        load_metrics.total_load_ns as f64 / 1_000_000.0
    ));
    write_state(&state_file, &record)?;
    info!(
        address = %bound_address,
        model = %app_state.model_id(),
        max_model_len = app_state.max_context_tokens(),
        max_queue_depth = options.max_queue_depth,
        queue_timeout_seconds = options.queue_timeout_seconds,
        tokenizer_ms,
        native_load_ms = load_metrics.total_load_ns as f64 / 1_000_000.0,
        "OpenAI-compatible service ready"
    );

    let shutdown_state = app_state.clone();
    let shutdown_signal = async move {
        tokio::select! {
            _ = shutdown_rx => {}
            _ = ctrl_c_or_pending() => {}
        }
        shutdown_state.begin_shutdown();
    };
    let server_result = axum::serve(listener, router(app_state))
        .with_graceful_shutdown(shutdown_signal)
        .await;
    record.status = if server_result.is_ok() {
        "stopped".to_owned()
    } else {
        "failed".to_owned()
    };
    record.stopped_unix_seconds = Some(epoch_seconds());
    record.message = server_result.as_ref().err().map(ToString::to_string);
    if let Err(error) = write_state(&state_file, &record) {
        error!(%error, "could not update stopped service state");
    }
    server_result.context("HTTP server failed")
}

pub fn start(options: StartOptions) -> Result<()> {
    let state_file = options
        .serve
        .state_file
        .clone()
        .unwrap_or_else(default_state_file);
    if let Ok(record) = read_state(&state_file) {
        if health_matches_record(&record, Duration::from_secs(2)) {
            bail!(
                "service is already ready at http://{} (pid {})",
                record.address,
                record.pid
            );
        }
    }
    if let Some(parent) = state_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create state directory {}", parent.display()))?;
    }
    let log_file = options
        .log_file
        .clone()
        .unwrap_or_else(|| state_file.with_file_name("service.log"));
    if let Some(parent) = log_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create log directory {}", parent.display()))?;
    }
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .with_context(|| format!("could not open log file {}", log_file.display()))?;
    let stderr = stdout.try_clone()?;
    let executable = std::env::current_exe().context("could not locate the qrt executable")?;
    let mut command = Command::new(executable);
    command.arg("serve");
    append_serve_args(&mut command, &options.serve, &state_file);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    configure_detached_process(&mut command);
    let mut child = command
        .spawn()
        .context("could not start detached qrt service")?;
    let deadline = Instant::now() + Duration::from_secs(options.wait_seconds);
    loop {
        if let Some(status) = child.try_wait()? {
            bail!(
                "service process exited before readiness with {status}; see {}",
                log_file.display()
            );
        }
        if let Ok(record) = read_state(&state_file) {
            if record.pid == child.id()
                && record.status == "ready"
                && health_matches_record(&record, Duration::from_secs(2))
            {
                println!(
                    "qrt ready: http://{} (pid {}, model {}, log {})",
                    record.address,
                    record.pid,
                    record.model,
                    log_file.display()
                );
                return Ok(());
            }
            if record.pid == child.id() && record.status == "failed" {
                bail!(
                    "service failed during startup: {}; see {}",
                    record.message.unwrap_or_default(),
                    log_file.display()
                );
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "service did not become ready within {} seconds (pid {}); it remains running, see {}",
                options.wait_seconds,
                child.id(),
                log_file.display()
            );
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

pub fn status(options: StatusOptions) -> Result<()> {
    let state_file = options.state_file.unwrap_or_else(default_state_file);
    let mut record = read_state(&state_file)
        .with_context(|| format!("no service state at {}", state_file.display()))?;
    let reachable = health_matches_record(&record, Duration::from_secs(2));
    if record.status == "ready" && !reachable {
        record.status = "unreachable".to_owned();
        record.message = Some("health endpoint is not reachable".to_owned());
    }
    let output = serde_json::to_string_pretty(&json_status(&record, reachable))?;
    println!("{output}");
    if !reachable {
        bail!("qrt service is not reachable")
    }
    Ok(())
}

pub fn stop(options: StopOptions) -> Result<()> {
    let state_file = options.state_file.unwrap_or_else(default_state_file);
    let record = read_state(&state_file)
        .with_context(|| format!("no service state at {}", state_file.display()))?;
    match health_identity(&record.address, Duration::from_secs(2)) {
        None => {
            println!(
                "qrt is already stopped or unreachable (last pid {})",
                record.pid
            );
            return Ok(());
        }
        Some(identity) if !identity_matches_record(&identity, &record) => {
            bail!(
                "refusing to stop endpoint http://{}: state expects pid {} model {}, health reports pid {} model {}",
                record.address,
                record.pid,
                record.model,
                identity.pid,
                identity.model
            );
        }
        Some(_) => {}
    }
    shutdown_request(
        &record.address,
        options.api_key.as_deref(),
        Duration::from_secs(5),
    )?;
    let deadline = Instant::now() + Duration::from_secs(options.wait_seconds);
    loop {
        let reachable = health_matches_record(&record, Duration::from_millis(750));
        let stopped = read_state(&state_file)
            .ok()
            .is_some_and(|latest| stopped_record_matches(&latest, &record));
        if !reachable && stopped {
            println!("qrt stopped (pid {})", record.pid);
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "service accepted shutdown but did not confirm complete resource release within {} seconds (reachable={reachable}, stopped_state={stopped})",
                options.wait_seconds,
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn append_serve_args(command: &mut Command, options: &ServeOptions, state_file: &Path) {
    if let Some(model) = &options.model {
        command.arg("--model").arg(model);
    }
    if let Some(provider) = &options.provider {
        command.arg("--provider").arg(provider);
    }
    if let Some(provider) = &options.arbitrary_moe_provider {
        command.arg("--arbitrary-moe-provider").arg(provider);
    }
    if let Some(kernel_dir) = &options.arbitrary_moe_kernel_dir {
        command.arg("--arbitrary-moe-kernel-dir").arg(kernel_dir);
    }
    if let Some(root) = &options.smooth_tail_moe_root {
        command.arg("--smooth-tail-moe-root").arg(root);
    }
    if let Some(env_file) = &options.env_file {
        command.arg("--env-file").arg(env_file);
    }
    for assignment in &options.env_overrides {
        command.arg("--set-env").arg(assignment);
    }
    command
        .arg("--model-id")
        .arg(&options.model_id)
        .arg("--host")
        .arg(options.host.to_string())
        .arg("--port")
        .arg(options.port.to_string())
        .arg("--max-model-len")
        .arg(options.max_model_len.to_string())
        .arg("--max-queue-depth")
        .arg(options.max_queue_depth.to_string())
        .arg("--queue-timeout-seconds")
        .arg(options.queue_timeout_seconds.to_string())
        .arg("--state-file")
        .arg(state_file);
    if let Some(api_key) = &options.api_key {
        command.env("QRT_API_KEY", api_key);
    }
    if options.allow_unauthenticated {
        command.arg("--allow-unauthenticated");
    }
}

fn resolve_required_path(
    explicit: Option<PathBuf>,
    env_name: &str,
    message: &str,
) -> Result<PathBuf> {
    explicit
        .or_else(|| std::env::var_os(env_name).map(PathBuf::from))
        .ok_or_else(|| anyhow!(message.to_owned()))
}

fn configure_arbitrary_moe_provider(
    explicit_provider: Option<PathBuf>,
    explicit_kernel_dir: Option<PathBuf>,
) -> Result<(PathBuf, PathBuf)> {
    let provider = explicit_provider.or_else(|| {
        std::env::var_os("QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_DLL").map(PathBuf::from)
    });
    let kernel_dir = explicit_kernel_dir.or_else(|| {
        std::env::var_os("QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_KERNEL_DIR").map(PathBuf::from)
    });
    let provider = provider.ok_or_else(|| {
        anyhow!(concat!(
            "--arbitrary-moe-provider is required for continuous batch-one ",
            "prompt lengths (or set ",
            "QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_DLL)"
        ))
    })?;
    let kernel_dir = kernel_dir.ok_or_else(|| {
        anyhow!(concat!(
            "--arbitrary-moe-kernel-dir is required for continuous batch-one ",
            "prompt lengths (or set ",
            "QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_KERNEL_DIR)"
        ))
    })?;
    if !provider.is_file() {
        bail!(
            "arbitrary-prompt q1024 MoE DLL does not exist: {}",
            provider.display()
        );
    }
    if !kernel_dir.is_dir() {
        bail!(
            "arbitrary-prompt q1024 MoE kernel directory does not exist: {}",
            kernel_dir.display()
        );
    }
    std::env::set_var("QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_PROVIDER", "1");
    std::env::set_var("QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_DLL", &provider);
    std::env::set_var(
        "QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_KERNEL_DIR",
        &kernel_dir,
    );
    Ok((provider, kernel_dir))
}

fn resolve_smooth_tail_moe_layout(
    explicit_root: Option<PathBuf>,
    env_file: Option<&Path>,
) -> Result<Option<SmoothTailMoeLayout>> {
    let requested_root = explicit_root
        .or_else(|| std::env::var_os("QRT_QWEN36_SMOOTH_TAIL_MOE_ROOT").map(PathBuf::from));
    let root = if let Some(root) = requested_root {
        root
    } else if let Some(root) = env_file
        .and_then(Path::parent)
        .map(|parent| parent.join("smooth-tail"))
    {
        if !root.exists() {
            return Ok(None);
        }
        root
    } else {
        return Ok(None);
    };
    if !root.is_dir() {
        bail!(
            "smooth-tail selected-MoE root does not exist or is not a directory: {}",
            root.display()
        );
    }

    let mut bindings = Vec::with_capacity(SMOOTH_TAIL_MOE_TOKEN_COUNTS.len());
    for tokens in SMOOTH_TAIL_MOE_TOKEN_COUNTS {
        let kernel_dir = root.join(format!("q{tokens}"));
        let provider = kernel_dir.join(format!("qrt_triton_moe_q{tokens}_fast_tail_provider.dll"));
        if !provider.is_file() {
            bail!(
                "smooth-tail q{tokens} provider is missing: {}",
                provider.display()
            );
        }
        if !kernel_dir.join("metadata.json").is_file() {
            bail!(
                "smooth-tail q{tokens} metadata is missing under {}",
                kernel_dir.display()
            );
        }
        for kernel in [
            "route_count",
            "route_prefix_by_program",
            "route_padded_prefix",
            "route_scatter",
            "gate_up_silu",
            "down",
        ] {
            let path = kernel_dir.join(format!("q{tokens}_selected_moe_{kernel}.hsaco"));
            if !path.is_file() {
                bail!(
                    "smooth-tail q{tokens} kernel is missing: {}",
                    path.display()
                );
            }
        }
        bindings.push(SmoothTailMoeBinding {
            tokens,
            provider,
            kernel_dir,
        });
    }
    Ok(Some(SmoothTailMoeLayout { root, bindings }))
}

fn configure_smooth_tail_moe_providers(
    explicit_root: Option<PathBuf>,
    env_file: Option<&Path>,
) -> Result<Option<PathBuf>> {
    let Some(layout) = resolve_smooth_tail_moe_layout(explicit_root, env_file)? else {
        return Ok(None);
    };
    if std::env::var_os("QRT_QWEN36_SMOOTH_TAIL_MOE_ROOT").is_none() {
        std::env::set_var("QRT_QWEN36_SMOOTH_TAIL_MOE_ROOT", &layout.root);
    }
    if std::env::var_os("QRT_QWEN36_SMOOTH_TAIL_MOE_PROVIDER").is_none() {
        std::env::set_var("QRT_QWEN36_SMOOTH_TAIL_MOE_PROVIDER", "1");
    }
    for binding in &layout.bindings {
        let dll_env = format!("QRT_QWEN36_SMOOTH_TAIL_Q{}_MOE_DLL", binding.tokens);
        let kernel_dir_env = format!("QRT_QWEN36_SMOOTH_TAIL_Q{}_MOE_KERNEL_DIR", binding.tokens);
        if std::env::var_os(&dll_env).is_none() {
            std::env::set_var(dll_env, &binding.provider);
        }
        if std::env::var_os(&kernel_dir_env).is_none() {
            std::env::set_var(kernel_dir_env, &binding.kernel_dir);
        }
    }
    Ok(Some(layout.root))
}

pub fn load_env_file(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("could not read runtime env file {}", path.display()))?;
    let mut count = 0;
    for (line_index, original) in content.lines().enumerate() {
        let mut line = original.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim_start();
        }
        let (key, value) = parse_env_assignment(line).with_context(|| {
            format!("invalid env entry at {}:{}", path.display(), line_index + 1)
        })?;
        std::env::set_var(key, value);
        count += 1;
    }
    Ok(count)
}

fn apply_env_overrides(assignments: &[String]) -> Result<usize> {
    for (index, assignment) in assignments.iter().enumerate() {
        let (key, value) = parse_env_assignment(assignment)
            .with_context(|| format!("invalid --set-env value at index {}", index + 1))?;
        std::env::set_var(key, value);
    }
    Ok(assignments.len())
}

fn parse_env_assignment(assignment: &str) -> Result<(&str, String)> {
    let Some((key, raw_value)) = assignment.split_once('=') else {
        bail!("expected KEY=VALUE");
    };
    if key.is_empty()
        || !key
            .bytes()
            .all(|value| value == b'_' || value.is_ascii_alphanumeric())
    {
        bail!("invalid environment key");
    }
    Ok((key, unquote_env_value(raw_value.trim())?))
}

fn unquote_env_value(value: &str) -> Result<String> {
    if value.len() >= 2 {
        let first = value.as_bytes()[0];
        let last = value.as_bytes()[value.len() - 1];
        if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
            return Ok(value[1..value.len() - 1].to_owned());
        }
        if first == b'\'' || first == b'"' || last == b'\'' || last == b'"' {
            bail!("unbalanced quote");
        }
    }
    Ok(value.to_owned())
}

fn write_state(path: &Path, record: &ServiceRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create state directory {}", parent.display()))?;
    }
    let data = serde_json::to_vec_pretty(record)?;
    fs::write(path, data)
        .with_context(|| format!("could not write service state {}", path.display()))
}

fn read_state(path: &Path) -> Result<ServiceRecord> {
    let data = fs::read(path)?;
    serde_json::from_slice(&data).context("invalid service state JSON")
}

pub fn default_state_file() -> PathBuf {
    #[cfg(windows)]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        return base.join("qrt").join("service.json");
    }
    #[cfg(not(windows))]
    {
        let base = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".local").join("state"))
            })
            .unwrap_or_else(std::env::temp_dir);
        base.join("qrt").join("service.json")
    }
}

fn health_identity(address: &str, timeout: Duration) -> Option<HealthIdentity> {
    let response = http_request(address, "GET", "/health", None, timeout).ok()?;
    parse_health_response(&response)
}

fn parse_health_response(response: &str) -> Option<HealthIdentity> {
    let (headers, body) = response.split_once("\r\n\r\n")?;
    let status_line = headers.lines().next()?;
    if !status_line.starts_with("HTTP/1.1 200") && !status_line.starts_with("HTTP/1.0 200") {
        return None;
    }
    serde_json::from_str(body.trim()).ok()
}

fn identity_matches_record(identity: &HealthIdentity, record: &ServiceRecord) -> bool {
    identity.ready && identity.pid == record.pid && identity.model == record.model
}

fn stopped_record_matches(latest: &ServiceRecord, expected: &ServiceRecord) -> bool {
    latest.pid == expected.pid
        && latest.address == expected.address
        && latest.model == expected.model
        && latest.status == "stopped"
        && latest.stopped_unix_seconds.is_some()
}

fn health_matches_record(record: &ServiceRecord, timeout: Duration) -> bool {
    health_identity(&record.address, timeout)
        .is_some_and(|identity| identity_matches_record(&identity, record))
}

fn shutdown_request(address: &str, api_key: Option<&str>, timeout: Duration) -> Result<()> {
    let response = http_request(address, "POST", "/admin/shutdown", api_key, timeout)?;
    if !response.starts_with("HTTP/1.1 200") && !response.starts_with("HTTP/1.0 200") {
        let status_line = response.lines().next().unwrap_or("invalid response");
        bail!("shutdown request failed: {status_line}");
    }
    Ok(())
}

fn http_request(
    address: &str,
    method: &str,
    path: &str,
    api_key: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let socket: SocketAddr = address.parse().context("invalid service address")?;
    let mut stream = TcpStream::connect_timeout(&socket, timeout)
        .with_context(|| format!("could not connect to {address}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Length: 0\r\n"
    );
    if let Some(api_key) = api_key {
        request.push_str("Authorization: Bearer ");
        request.push_str(api_key);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes())?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn json_status(record: &ServiceRecord, reachable: bool) -> serde_json::Value {
    serde_json::json!({
        "status": record.status,
        "reachable": reachable,
        "endpoint": format!("http://{}", record.address),
        "pid": record.pid,
        "model": record.model,
        "model_path": record.model_path,
        "provider_dll": record.provider_dll,
        "arbitrary_moe_provider_dll": record.arbitrary_moe_provider_dll,
        "arbitrary_moe_kernel_dir": record.arbitrary_moe_kernel_dir,
        "smooth_tail_moe_root": record.smooth_tail_moe_root,
        "max_model_len": record.max_model_len,
        "max_queue_depth": record.max_queue_depth,
        "queue_timeout_seconds": record.queue_timeout_seconds,
        "repo_commit": record.repo_commit,
        "host": record.host,
        "started_unix_seconds": record.started_unix_seconds,
        "ready_unix_seconds": record.ready_unix_seconds,
        "stopped_unix_seconds": record.stopped_unix_seconds,
        "message": record.message,
    })
}

fn default_record_max_queue_depth() -> usize {
    DEFAULT_MAX_QUEUE_DEPTH
}

fn default_record_queue_timeout_seconds() -> u64 {
    DEFAULT_QUEUE_TIMEOUT_SECONDS
}

fn machine_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn ctrl_c_or_pending() {
    if tokio::signal::ctrl_c().await.is_err() {
        std::future::pending::<()>().await;
    }
}

#[cfg(windows)]
fn configure_detached_process(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    command.creation_flags(CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
}

#[cfg(unix)]
fn configure_detached_process(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(any(windows, unix)))]
fn configure_detached_process(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_value_unquoting_is_bounded() {
        assert_eq!(unquote_env_value("abc").unwrap(), "abc");
        assert_eq!(unquote_env_value("\"a b\"").unwrap(), "a b");
        assert!(unquote_env_value("\"broken").is_err());
        assert_eq!(
            parse_env_assignment("QRT_TEST_VALUE='a b'").unwrap(),
            ("QRT_TEST_VALUE", "a b".to_owned())
        );
        assert!(parse_env_assignment("NOT-VALID=1").is_err());
        assert!(parse_env_assignment("MISSING_EQUALS").is_err());
    }

    #[test]
    fn health_response_binds_pid_and_model_identity() {
        let response = concat!(
            "HTTP/1.1 200 OK\r\n",
            "content-type: application/json\r\n",
            "content-length: 74\r\n\r\n",
            r#"{"status":"ok","ready":true,"pid":42,"model":"test-model"}"#
        );
        let identity = parse_health_response(response).unwrap();
        let record = ServiceRecord {
            schema_version: 1,
            status: "ready".to_owned(),
            pid: 42,
            address: "127.0.0.1:8000".to_owned(),
            model: "test-model".to_owned(),
            model_path: "model".to_owned(),
            provider_dll: "provider".to_owned(),
            arbitrary_moe_provider_dll: Some("q1024-provider".to_owned()),
            arbitrary_moe_kernel_dir: Some("q1024-kernels".to_owned()),
            smooth_tail_moe_root: Some("smooth-tail".to_owned()),
            max_model_len: 262_144,
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            queue_timeout_seconds: DEFAULT_QUEUE_TIMEOUT_SECONDS,
            repo_commit: "test".to_owned(),
            host: "test".to_owned(),
            started_unix_seconds: 1,
            ready_unix_seconds: Some(2),
            stopped_unix_seconds: None,
            message: None,
        };
        assert!(identity_matches_record(&identity, &record));
        let mut wrong_pid = record.clone();
        wrong_pid.pid += 1;
        assert!(!identity_matches_record(&identity, &wrong_pid));
    }

    #[test]
    fn stopped_state_binds_the_completed_service_instance() {
        let expected = ServiceRecord {
            schema_version: 1,
            status: "ready".to_owned(),
            pid: 42,
            address: "127.0.0.1:8000".to_owned(),
            model: "test-model".to_owned(),
            model_path: "model".to_owned(),
            provider_dll: "provider".to_owned(),
            arbitrary_moe_provider_dll: Some("q1024-provider".to_owned()),
            arbitrary_moe_kernel_dir: Some("q1024-kernels".to_owned()),
            smooth_tail_moe_root: Some("smooth-tail".to_owned()),
            max_model_len: 262_144,
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            queue_timeout_seconds: DEFAULT_QUEUE_TIMEOUT_SECONDS,
            repo_commit: "test".to_owned(),
            host: "test".to_owned(),
            started_unix_seconds: 1,
            ready_unix_seconds: Some(2),
            stopped_unix_seconds: None,
            message: None,
        };
        let mut stopped = expected.clone();
        stopped.status = "stopped".to_owned();
        stopped.stopped_unix_seconds = Some(3);
        assert!(stopped_record_matches(&stopped, &expected));

        let mut wrong_pid = stopped.clone();
        wrong_pid.pid += 1;
        assert!(!stopped_record_matches(&wrong_pid, &expected));
        let mut still_ready = stopped;
        still_ready.status = "ready".to_owned();
        assert!(!stopped_record_matches(&still_ready, &expected));
    }

    #[test]
    fn packaged_smooth_tail_layout_requires_every_provider_and_kernel() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "qrt-smooth-tail-layout-{}-{nonce}",
            std::process::id()
        ));
        for tokens in SMOOTH_TAIL_MOE_TOKEN_COUNTS {
            let dir = root.join(format!("q{tokens}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join(format!("qrt_triton_moe_q{tokens}_fast_tail_provider.dll")),
                b"provider",
            )
            .unwrap();
            fs::write(dir.join("metadata.json"), b"{}").unwrap();
            for kernel in [
                "route_count",
                "route_prefix_by_program",
                "route_padded_prefix",
                "route_scatter",
                "gate_up_silu",
                "down",
            ] {
                fs::write(
                    dir.join(format!("q{tokens}_selected_moe_{kernel}.hsaco")),
                    b"kernel",
                )
                .unwrap();
            }
        }

        let layout = resolve_smooth_tail_moe_layout(Some(root.clone()), None)
            .unwrap()
            .unwrap();
        assert_eq!(layout.root, root);
        assert_eq!(layout.bindings.len(), SMOOTH_TAIL_MOE_TOKEN_COUNTS.len());
        let missing = root.join("q256/q256_selected_moe_down.hsaco");
        fs::remove_file(&missing).unwrap();
        let error = resolve_smooth_tail_moe_layout(Some(root.clone()), None).unwrap_err();
        assert!(error
            .to_string()
            .contains("smooth-tail q256 kernel is missing"));
        fs::remove_dir_all(root).unwrap();
    }
}
