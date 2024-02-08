#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod embeddings;
use std::ffi::OsStr;

use candle_core::Tensor;
use clap::{Parser, Subcommand};

use human_panic::setup_panic;

use ignore::WalkBuilder;
use itertools::Itertools;
use thiserror::Error;
use tokio::{fs::File, io::AsyncReadExt};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt, Registry};
mod markdown_parser;
use itertools::Either::{Left, Right};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Input folder to parse.
    #[arg(long, default_value = ".")]
    input_folder: String,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Batch size
    #[arg(short, long, default_value_t = 10)]
    batch_size: i32,

    /// Path to the tokenizer file.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Path to the model file.
    #[arg(long)]
    model: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// generate embeddings
    Embed {
        // /// lists test values
        // #[arg(short, long)]
        // list: bool,
    },
}

#[derive(Debug, Clone)]
struct MarkdownFile {
    file_name: String,
    content: String,
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

    let cli = Cli::parse();
    let _guard = if cli.tracing {
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
    let walk = WalkBuilder::new(cli.input_folder)
        .hidden(true)
        .filter_entry(|entry| {
            entry.path().is_dir()
                || entry
                    .path()
                    .extension()
                    .and_then(OsStr::to_str)
                    .map(|extension| extension.eq("md"))
                    .unwrap_or(false)
        })
        .build();
    let walk = walk.filter_map(|entry| entry.ok().filter(|dir_entry| dir_entry.path().is_file()));

    let mut markdown_files: Vec<MarkdownFile> = vec![];
    for result in walk {
        let path = result.path();

        let mut f = File::open(path).await?;
        let mut buffer = String::new();
        f.read_to_string(&mut buffer).await?;

        let markdown = buffer.as_str();
        // let mdast = markdown::to_mdast(markdown, &markdown::ParseOptions::default())
        //     .map_err(ProcessError::InvalidMarkdown)?;
        markdown_files.push(MarkdownFile {
            file_name: path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned().to_string())
                .unwrap_or_default(),
            content: markdown.to_string(),
        });
        // println!("{:?}", mdast);
    }

    let markdown_files = markdown_files
        .into_iter()
        .take(30)
        .collect::<Vec<MarkdownFile>>();
    let (model, mut tokenizer) =
        embeddings::build_model_and_tokenizer(cli.model, cli.tokenizer, cli.cpu).await?;

    let contents = &markdown_files
        .iter()
        .map(|md| md.content.clone())
        .collect_vec();
    let tokens = tokenizer
        .encode_batch(contents.clone(), false)
        .map_err(|err| anyhow::anyhow!("Failed to tokenize batch."))?;
    println!(
        "#files: {:?}",
        &markdown_files
            .iter()
            .map(|f| f.file_name.clone())
            .collect_vec()
    );

    const NUM_EMBEDDINGS: usize = 8192;

    let files = &markdown_files
        .iter()
        .zip(tokens.iter())
        .flat_map(move |(md, encoding)| {
            encoding
                .get_ids()
                .to_vec()
                .chunks(NUM_EMBEDDINGS)
                .into_iter()
                .map(|res| (md, res.to_vec()))
                .collect_vec()
        })
        .collect_vec();

    files
        .iter()
        .for_each(|res| println!("{:?} - {:?}", res.0.file_name, res.1.len()));

    let files = files.iter().take(4).collect_vec();
    let token_ids_batch = &files.iter().map(|tuple| tuple.1.clone()).collect_vec();
    let embeddings = embeddings::embed(
        &model,
        &mut tokenizer,
        false,
        NUM_EMBEDDINGS,
        token_ids_batch,
    )?;

    let results = files
        .iter()
        .map(move |(file, _)| *file)
        .zip(embeddings)
        .collect_vec();

    results.iter().for_each(|result| {
        let file_name = (*result.0).file_name.clone();
        match &result.1 {
            Left(embedding) => {
                let first_ten = embedding.1.iter().take(10).collect_vec();
                println!("{:} {:?}", file_name, first_ten)
            }
            Right(err) => {
                println!("{:} failed", file_name)
            }
        }
    });
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}
