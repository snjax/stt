use std::{
    env,
    fmt::Display,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    session::Session,
    value::TensorRef,
};

use crate::{
    audio::load_wav_mono_16k,
    mel::{MelSpectrogram, N_MELS, log_mel_spectrogram},
    tokenizer::Tokenizer,
};

const BLANK_ID: i64 = 1024;
const ENCODER_DIM: usize = 768;
const DECODER_DIM: usize = 320;
const MAX_SYMBOLS_PER_STEP: usize = 10;

pub struct ModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joint: PathBuf,
    pub tokenizer: PathBuf,
}

pub struct Transcriber {
    encoder: Session,
    decoder: Session,
    joint: Session,
    tokenizer: Tokenizer,
}

struct EncoderOutput {
    encoded: Vec<f32>,
    shape: Vec<usize>,
    encoded_len: usize,
}

impl ModelPaths {
    pub fn discover() -> Result<Self> {
        for candidate in candidate_model_dirs() {
            let model_paths = Self {
                encoder: candidate.join("v3_e2e_rnnt_encoder.onnx"),
                decoder: candidate.join("v3_e2e_rnnt_decoder.onnx"),
                joint: candidate.join("v3_e2e_rnnt_joint.onnx"),
                tokenizer: candidate.join("tokenizer.model"),
            };

            if model_paths.encoder.is_file()
                && model_paths.decoder.is_file()
                && model_paths.joint.is_file()
                && model_paths.tokenizer.is_file()
            {
                return Ok(model_paths);
            }
        }

        bail!(
            "unable to find models/onnx directory with encoder, decoder, joint, and tokenizer.model"
        )
    }
}

impl Transcriber {
    pub fn new(paths: ModelPaths) -> Result<Self> {
        init_ort()?;

        let tokenizer = Tokenizer::open(&paths.tokenizer)?;
        eprintln!("Loading encoder...");
        let encoder = build_session(&paths.encoder)?;
        eprintln!("Loading decoder...");
        let decoder = build_session(&paths.decoder)?;
        eprintln!("Loading joint...");
        let joint = build_session(&paths.joint)?;

        if tokenizer.blank_id() as i64 != BLANK_ID {
            bail!(
                "tokenizer vocabulary size {} does not match expected blank id {}",
                tokenizer.blank_id(),
                BLANK_ID
            );
        }

        Ok(Self {
            encoder,
            decoder,
            joint,
            tokenizer,
        })
    }

    pub fn transcribe_wav_file(&mut self, path: &Path) -> Result<String> {
        let samples = load_wav_mono_16k(path)?;
        self.transcribe_samples(&samples)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mel = log_mel_spectrogram(samples);
        let encoder_output = self.run_encoder(&mel)?;
        let token_ids = self.greedy_decode(&encoder_output)?;
        self.tokenizer.decode(&token_ids)
    }

    fn run_encoder(&mut self, mel: &MelSpectrogram) -> Result<EncoderOutput> {
        let lengths = [mel.frames as i64];
        let audio_signal =
            TensorRef::from_array_view(([1_usize, N_MELS, mel.frames], &mel.data[..]))
                .map_err(ort_error)?;
        let audio_length =
            TensorRef::from_array_view(([1_usize], &lengths[..])).map_err(ort_error)?;
        let mut outputs = self
            .encoder
            .run(ort::inputs! {
                "audio_signal" => audio_signal,
                "length" => audio_length,
            })
            .map_err(ort_error)?;

        let encoded_value = outputs
            .remove("encoded")
            .context("encoder output `encoded` missing")?;
        let encoded_len_value = outputs
            .remove("encoded_len")
            .context("encoder output `encoded_len` missing")?;

        let encoded = encoded_value
            .try_extract_array::<f32>()
            .map_err(ort_error)?;
        let shape = encoded.shape().to_vec();
        let encoded_vec = encoded.iter().copied().collect::<Vec<_>>();

        // encoded_len may be i64 or i32 depending on ORT version
        let encoded_len = if let Ok(arr) = encoded_len_value.try_extract_array::<i64>() {
            *arr.iter().next().ok_or_else(|| anyhow!("encoder returned empty encoded_len"))? as usize
        } else {
            let arr = encoded_len_value.try_extract_array::<i32>().map_err(ort_error)?;
            *arr.iter().next().ok_or_else(|| anyhow!("encoder returned empty encoded_len"))? as usize
        };

        Ok(EncoderOutput {
            encoded: encoded_vec,
            shape,
            encoded_len,
        })
    }

    fn greedy_decode(&mut self, encoder_output: &EncoderOutput) -> Result<Vec<u32>> {
        let mut emitted = Vec::new();
        let mut prev_token = BLANK_ID;
        let mut h = vec![0.0_f32; DECODER_DIM];
        let mut c = vec![0.0_f32; DECODER_DIM];

        for time_index in 0..encoder_output.encoded_len {
            let frame = select_frame(encoder_output, time_index)?;
            let mut symbols = 0;

            while symbols < MAX_SYMBOLS_PER_STEP {
                let label = [prev_token];
                let x = TensorRef::from_array_view(([1_usize, 1_usize], &label[..]))
                    .map_err(ort_error)?;
                let h_input = TensorRef::from_array_view(([1_usize, 1_usize, DECODER_DIM], &h[..]))
                    .map_err(ort_error)?;
                let c_input = TensorRef::from_array_view(([1_usize, 1_usize, DECODER_DIM], &c[..]))
                    .map_err(ort_error)?;
                let mut decoder_outputs = self
                    .decoder
                    .run(ort::inputs! {
                        "x" => x,
                        "h.1" => h_input,
                        "c.1" => c_input,
                    })
                    .map_err(ort_error)?;

                let dec_value = decoder_outputs
                    .remove("dec")
                    .context("decoder output `dec` missing")?;
                let h_value = decoder_outputs
                    .remove("h")
                    .context("decoder output `h` missing")?;
                let c_value = decoder_outputs
                    .remove("c")
                    .context("decoder output `c` missing")?;

                let dec = dec_value
                    .try_extract_array::<f32>()
                    .map_err(ort_error)?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let h_new = h_value
                    .try_extract_array::<f32>()
                    .map_err(ort_error)?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let c_new = c_value
                    .try_extract_array::<f32>()
                    .map_err(ort_error)?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();

                let enc = TensorRef::from_array_view(([1_usize, ENCODER_DIM, 1_usize], &frame[..]))
                    .map_err(ort_error)?;
                let dec = TensorRef::from_array_view(([1_usize, DECODER_DIM, 1_usize], &dec[..]))
                    .map_err(ort_error)?;
                let mut joint_outputs = self
                    .joint
                    .run(ort::inputs! {
                        "enc" => enc,
                        "dec" => dec,
                    })
                    .map_err(ort_error)?;

                let joint_value = joint_outputs
                    .remove("joint")
                    .context("joint output `joint` missing")?;
                let joint = joint_value.try_extract_array::<f32>().map_err(ort_error)?;

                let token = joint
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|left, right| {
                        left.1
                            .partial_cmp(&right.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(index, _)| index as i64)
                    .context("joint output was empty")?;

                if token == BLANK_ID {
                    break;
                }

                emitted.push(token as u32);
                prev_token = token;
                h = h_new;
                c = c_new;
                symbols += 1;
            }
        }

        Ok(emitted)
    }
}

fn find_ort_dylib() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Homebrew on macOS (Apple Silicon)
    #[cfg(target_os = "macos")]
    {
        let homebrew_path = PathBuf::from("/opt/homebrew/lib/libonnxruntime.dylib");
        if homebrew_path.exists() {
            return Some(homebrew_path);
        }
        // Intel Mac
        let homebrew_intel = PathBuf::from("/usr/local/lib/libonnxruntime.dylib");
        if homebrew_intel.exists() {
            return Some(homebrew_intel);
        }
    }

    // Linux standard paths
    #[cfg(target_os = "linux")]
    {
        let paths = [
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
            "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
        ];
        for p in &paths {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

fn init_ort() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

    let result = INIT.get_or_init(|| {
        // Find and load the ORT dylib
        let dylib_path = find_ort_dylib()
            .ok_or_else(|| "Could not find libonnxruntime. Install via: brew install onnxruntime (macOS) or set ORT_DYLIB_PATH".to_string())?;

        eprintln!("Loading ORT from: {}", dylib_path.display());

        ort::init_from(dylib_path.to_string_lossy().to_string())
            .with_name("stt")
            .commit()
            .map_err(|e| format!("{e}"))?;

        Ok(())
    });

    result
        .as_ref()
        .map(|_| ())
        .map_err(|error| anyhow!(error.clone()))
}

fn build_session(path: &Path) -> Result<Session> {
    let session = Session::builder()
        .map_err(ort_error)?
        .with_intra_threads(1)
        .map_err(ort_error)?
        .with_execution_providers([
            #[cfg(target_os = "macos")]
            CoreMLExecutionProvider::default().build(),
            #[cfg(target_os = "linux")]
            ort::execution_providers::CUDAExecutionProvider::default().build(),
        ])
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)?;
    Ok(session)
}

fn select_frame(output: &EncoderOutput, time_index: usize) -> Result<Vec<f32>> {
    if output.shape.len() != 3 || output.shape[0] != 1 {
        bail!("unexpected encoder output shape: {:?}", output.shape);
    }

    let time_first = output.shape[1] == output.encoded_len && output.shape[2] == ENCODER_DIM;
    let channel_first = output.shape[1] == ENCODER_DIM && output.shape[2] == output.encoded_len;

    let mut frame = vec![0.0_f32; ENCODER_DIM];
    if time_first {
        let stride = output.shape[2];
        let start = time_index * stride;
        frame.copy_from_slice(&output.encoded[start..start + stride]);
        return Ok(frame);
    }

    if channel_first {
        let time_stride = output.shape[2];
        for channel in 0..ENCODER_DIM {
            frame[channel] = output.encoded[channel * time_stride + time_index];
        }
        return Ok(frame);
    }

    bail!(
        "unable to map encoder output shape {:?} to [1, time, 768] or [1, 768, time]",
        output.shape
    )
}

fn candidate_model_dirs() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(custom_dir) = env::var("STT_MODEL_DIR") {
        let custom_dir = PathBuf::from(custom_dir);
        candidates.push(custom_dir.clone());
        candidates.push(custom_dir.join("models").join("onnx"));
    }

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("models").join("onnx"));
    }

    if let Ok(executable) = env::current_exe() {
        let mut cursor = executable.parent().map(Path::to_path_buf);
        while let Some(dir) = cursor {
            candidates.push(dir.join("models").join("onnx"));
            cursor = dir.parent().map(Path::to_path_buf);
        }
    }

    dedupe_paths(candidates)
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped: Vec<PathBuf> = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}

fn ort_error<E>(error: E) -> anyhow::Error
where
    E: Display,
{
    anyhow!("{error}")
}
