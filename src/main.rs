mod app;
mod audio;
mod inference;
mod mel;
mod tokenizer;
mod whisper;

use std::{
    env,
    path::{Path, PathBuf},
    process,
};

use anyhow::{Context, Result, bail};

use crate::{
    app::{BackendChoice, run_gui},
    inference::{ModelPaths, Transcriber},
    whisper::WhisperTranscriber,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:?}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let (backend, file_path) = parse_args()?;

    match file_path {
        Some(path) => transcribe_file(backend, &path),
        None => run_gui(backend),
    }
}

fn parse_args() -> Result<(BackendChoice, Option<PathBuf>)> {
    let mut backend = BackendChoice::GigaAm;
    let mut file_path: Option<PathBuf> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--whisper" => backend = BackendChoice::Whisper,
            "--file" => {
                let path = args
                    .next()
                    .map(PathBuf::from)
                    .context("missing path after --file")?;

                if file_path.replace(path).is_some() {
                    bail!("`--file` may only be provided once");
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => bail!("unknown argument: {other}. Supported usage: {}", usage()),
        }
    }

    Ok((backend, file_path))
}

fn transcribe_file(backend: BackendChoice, path: &Path) -> Result<()> {
    let text = match backend {
        BackendChoice::GigaAm => {
            let mut transcriber = Transcriber::new(ModelPaths::discover()?)?;
            transcriber.transcribe_wav_file(path)?
        }
        BackendChoice::Whisper => {
            let mut transcriber = WhisperTranscriber::new()?;
            transcriber.transcribe_wav_file(path)?
        }
    };

    println!("{text}");
    Ok(())
}

fn usage() -> &'static str {
    "cargo run -- [--whisper] [--file path/to.wav]"
}
