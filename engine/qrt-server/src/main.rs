use anyhow::Result;
use clap::{Parser, Subcommand};
use qrt_server::lifecycle::{self, ServeOptions, StartOptions, StatusOptions, StopOptions};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "qrt",
    version,
    about = "Native Qwen3.6 inference service for Windows AMD gfx1151"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the resident OpenAI-compatible HTTP service in the foreground.
    Serve(ServeOptions),
    /// Start the resident service as a detached background process.
    Start(StartOptions),
    /// Print service state and verify the health endpoint.
    Status(StatusOptions),
    /// Gracefully stop the resident model service.
    Stop(StopOptions),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    match Cli::parse().command {
        Command::Serve(options) => lifecycle::serve(options).await,
        Command::Start(options) => lifecycle::start(options),
        Command::Status(options) => lifecycle::status(options),
        Command::Stop(options) => lifecycle::stop(options),
    }
}
