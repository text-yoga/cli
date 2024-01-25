#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod embeddings;
use candle_core::Tensor;
use clap::Parser;

use human_panic::setup_panic;

use markdown;
use thiserror::Error;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt, Registry};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    model: Option<String>,
}

#[derive(Error, Debug)]
pub enum ProcessError {
    #[error("Failed to parse markdown: {0}")]
    InvalidMarkdown(String),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_panic!(Metadata {
        name: env!("CARGO_PKG_NAME").into(),
        version: env!("CARGO_PKG_VERSION").into(),
        authors: "Jan Schulte".into(),
        homepage: "https://github.com/text-yoga/cli/issues".into(),
    });
    // use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        // let subscriber = tracing_subscriber::FmtSubscriber::builder();
        // tracing_subscriber::registry().with(chrome_layer).init();

        let (non_blocking, _) = tracing_appender::non_blocking(std::io::stdout());
        let std_out = fmt::Layer::default().with_writer(non_blocking);

        let subscriber = Registry::default().with(chrome_layer).with(std_out).init();
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        // .with_max_level(Level::TRACE)
        // builds the subscriber.

        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    println!(
        "{:?}",
        markdown::to_mdast("# Hey, *you*!", &markdown::ParseOptions::default())
            .map_err(ProcessError::InvalidMarkdown)?
    );

    let (model, mut tokenizer) =
        embeddings::build_model_and_tokenizer(args.model, args.tokenizer, args.cpu).await?;

    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]
    .map(String::from)
    .to_vec();

    let result = embeddings::embed(&model, &mut tokenizer, false, sentences);
    println!("Result: {:#?}", result);
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}
