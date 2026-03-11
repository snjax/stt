use std::{
    env,
    ffi::{CStr, CString},
    path::{Path, PathBuf},
    ptr,
};

use anyhow::{Context, Result, bail};

// --- FFI bindings to whisper.cpp ---

#[repr(C)]
#[allow(non_camel_case_types)]
struct whisper_context {
    _opaque: [u8; 0],
}

#[repr(C)]
#[allow(non_camel_case_types)]
struct whisper_state {
    _opaque: [u8; 0],
}

// We don't need the full alignment_heads struct, just a zero-initialized placeholder
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct whisper_aheads {
    n_heads: usize,
    heads: *const u8, // Actually whisper_ahead*, but we don't use it
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct whisper_context_params {
    use_gpu: bool,
    flash_attn: bool,
    gpu_device: i32,
    dtw_token_timestamps: bool,
    dtw_aheads_preset: i32,
    dtw_n_top: i32,
    dtw_aheads: whisper_aheads,
    dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct whisper_vad_params {
    threshold: f32,
    min_speech_duration_ms: i32,
    min_silence_duration_ms: i32,
    max_speech_duration_s: f32,
    speech_pad_ms: i32,
    samples_overlap: f32,
}

#[repr(i32)]
#[allow(non_camel_case_types, dead_code)]
enum whisper_sampling_strategy {
    WHISPER_SAMPLING_GREEDY = 0,
    WHISPER_SAMPLING_BEAM_SEARCH = 1,
}

// Callback types (we use null pointers for all of them)
type WhisperNewSegmentCallback = Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, i32, *mut std::ffi::c_void)>;
type WhisperProgressCallback = Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, i32, *mut std::ffi::c_void)>;
type WhisperEncoderBeginCallback = Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, *mut std::ffi::c_void) -> bool>;
type WhisperLogitsFilterCallback = Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, *const u8, i32, *mut f32, *mut std::ffi::c_void)>;
type GgmlAbortCallback = Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> bool>;

#[repr(C)]
#[allow(non_camel_case_types)]
struct whisper_full_params {
    strategy: whisper_sampling_strategy,
    n_threads: i32,
    n_max_text_ctx: i32,
    offset_ms: i32,
    duration_ms: i32,

    translate: bool,
    no_context: bool,
    no_timestamps: bool,
    single_segment: bool,
    print_special: bool,
    print_progress: bool,
    print_realtime: bool,
    print_timestamps: bool,

    token_timestamps: bool,
    thold_pt: f32,
    thold_ptsum: f32,
    max_len: i32,
    split_on_word: bool,
    max_tokens: i32,

    debug_mode: bool,
    audio_ctx: i32,

    tdrz_enable: bool,

    suppress_regex: *const i8,

    initial_prompt: *const i8,
    carry_initial_prompt: bool,
    prompt_tokens: *const i32,
    prompt_n_tokens: i32,

    language: *const i8,
    detect_language: bool,

    suppress_blank: bool,
    suppress_nst: bool,

    temperature: f32,
    max_initial_ts: f32,
    length_penalty: f32,

    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,

    greedy: GreedyParams,
    beam_search: BeamSearchParams,

    new_segment_callback: WhisperNewSegmentCallback,
    new_segment_callback_user_data: *mut std::ffi::c_void,

    progress_callback: WhisperProgressCallback,
    progress_callback_user_data: *mut std::ffi::c_void,

    encoder_begin_callback: WhisperEncoderBeginCallback,
    encoder_begin_callback_user_data: *mut std::ffi::c_void,

    abort_callback: GgmlAbortCallback,
    abort_callback_user_data: *mut std::ffi::c_void,

    logits_filter_callback: WhisperLogitsFilterCallback,
    logits_filter_callback_user_data: *mut std::ffi::c_void,

    grammar_rules: *const *const u8,
    n_grammar_rules: usize,
    i_start_rule: usize,
    grammar_penalty: f32,

    vad: bool,
    vad_model_path: *const i8,
    vad_params: whisper_vad_params,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct GreedyParams {
    best_of: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct BeamSearchParams {
    beam_size: i32,
    patience: f32,
}

unsafe extern "C" {
    fn whisper_context_default_params() -> whisper_context_params;
    fn whisper_init_from_file_with_params(
        path_model: *const i8,
        params: whisper_context_params,
    ) -> *mut whisper_context;
    fn whisper_free(ctx: *mut whisper_context);

    fn whisper_full_default_params(strategy: whisper_sampling_strategy) -> whisper_full_params;
    fn whisper_full(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const f32,
        n_samples: i32,
    ) -> i32;
    fn whisper_full_n_segments(ctx: *mut whisper_context) -> i32;
    fn whisper_full_get_segment_text(ctx: *mut whisper_context, i_segment: i32) -> *const i8;

    fn whisper_pcm_to_mel(
        ctx: *mut whisper_context,
        samples: *const f32,
        n_samples: i32,
        n_threads: i32,
    ) -> i32;
    fn whisper_lang_auto_detect(
        ctx: *mut whisper_context,
        offset_ms: i32,
        n_threads: i32,
        lang_probs: *mut f32,
    ) -> i32;
    fn whisper_lang_max_id() -> i32;
    fn whisper_lang_str(id: i32) -> *const i8;
}

// --- Safe Rust wrapper ---

pub struct WhisperCppTranscriber {
    ctx: *mut whisper_context,
}

// whisper_context is thread-safe for sequential access from a single thread
unsafe impl Send for WhisperCppTranscriber {}

impl WhisperCppTranscriber {
    pub fn new() -> Result<Self> {
        let model_path = find_whisper_model()?;
        eprintln!("Loading whisper model from: {}", model_path.display());

        let c_path = CString::new(model_path.to_str().context("model path is not valid UTF-8")?)
            .context("model path contains null byte")?;

        let ctx = unsafe {
            let cparams = whisper_context_default_params();
            whisper_init_from_file_with_params(c_path.as_ptr(), cparams)
        };

        if ctx.is_null() {
            bail!("failed to load whisper model from {}", model_path.display());
        }

        eprintln!("Whisper model loaded successfully");
        Ok(Self { ctx })
    }

    pub fn transcribe_samples(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = unsafe {
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY)
        };

        let lang = CString::new("auto").unwrap();
        params.language = lang.as_ptr();
        params.no_context = true;
        params.single_segment = false;
        params.print_special = false;
        params.print_progress = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        params.n_threads = num_cpus().min(8) as i32;

        // Two-pass: detect language first, restrict to ru/en
        let detected = self.detect_language(samples)?;
        let forced_lang = if detected == "ru" { "ru" } else { "en" };
        let forced = CString::new(forced_lang).unwrap();
        params.language = forced.as_ptr();
        params.detect_language = false;

        let ret = unsafe {
            whisper_full(self.ctx, params, samples.as_ptr(), samples.len() as i32)
        };

        if ret != 0 {
            bail!("whisper_full failed with code {ret}");
        }

        let n_segments = unsafe { whisper_full_n_segments(self.ctx) };
        let mut text = String::new();
        for i in 0..n_segments {
            let segment_text = unsafe {
                let ptr = whisper_full_get_segment_text(self.ctx, i);
                if ptr.is_null() {
                    continue;
                }
                CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("")
            };
            text.push_str(segment_text);
        }

        Ok(text.trim().to_owned())
    }

    pub fn transcribe_samples_streaming(
        &self,
        samples: &[f32],
        single_segment: bool,
    ) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = unsafe {
            whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY)
        };

        let lang = CString::new("auto").unwrap();
        params.language = lang.as_ptr();
        params.no_context = true;
        params.single_segment = single_segment;
        params.print_special = false;
        params.print_progress = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        params.n_threads = num_cpus().min(8) as i32;

        // Two-pass: detect language first, restrict to ru/en
        let detected = self.detect_language(samples)?;
        let forced_lang = if detected == "ru" { "ru" } else { "en" };
        let forced = CString::new(forced_lang).unwrap();
        params.language = forced.as_ptr();
        params.detect_language = false;

        let ret = unsafe {
            whisper_full(self.ctx, params, samples.as_ptr(), samples.len() as i32)
        };

        if ret != 0 {
            bail!("whisper_full (streaming) failed with code {ret}");
        }

        let n_segments = unsafe { whisper_full_n_segments(self.ctx) };
        let mut text = String::new();
        for i in 0..n_segments {
            let segment_text = unsafe {
                let ptr = whisper_full_get_segment_text(self.ctx, i);
                if ptr.is_null() {
                    continue;
                }
                CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("")
            };
            text.push_str(segment_text);
        }

        Ok(text.trim().to_owned())
    }

    /// Detect language from audio samples, restricted to Russian and English.
    /// Returns "ru" or "en".
    fn detect_language(&self, samples: &[f32]) -> Result<String> {
        // Use at most first 30s for detection (whisper limit)
        let max_samples = 16_000 * 30;
        let detect_samples = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            samples
        };

        let n_threads = num_cpus().min(8) as i32;

        let ret = unsafe {
            whisper_pcm_to_mel(
                self.ctx,
                detect_samples.as_ptr(),
                detect_samples.len() as i32,
                n_threads,
            )
        };
        if ret != 0 {
            bail!("whisper_pcm_to_mel failed with code {ret}");
        }

        let max_id = unsafe { whisper_lang_max_id() };
        let mut lang_probs = vec![0.0f32; (max_id + 1) as usize];

        let detected_id = unsafe {
            whisper_lang_auto_detect(self.ctx, 0, n_threads, lang_probs.as_mut_ptr())
        };

        if detected_id < 0 {
            bail!("whisper_lang_auto_detect failed with code {detected_id}");
        }

        let detected = unsafe {
            let ptr = whisper_lang_str(detected_id);
            if ptr.is_null() {
                "en"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("en")
            }
        };

        let lang = if detected == "ru" { "ru" } else { "en" };
        eprintln!("Language detected: {detected} → using: {lang}");
        Ok(lang.to_owned())
    }

    pub fn transcribe_wav_file(&self, path: &Path) -> Result<String> {
        let samples = crate::audio::load_wav_mono_16k(path)?;
        self.transcribe_samples(&samples)
    }
}

impl Drop for WhisperCppTranscriber {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { whisper_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn find_whisper_model() -> Result<PathBuf> {
    if let Ok(path) = env::var("STT_WHISPER_MODEL") {
        let p = PathBuf::from(&path);
        if p.is_file() {
            return Ok(p);
        }
    }

    let model_names = [
        "ggml-large-v3-turbo.bin",
        "ggml-large-v3-turbo-q5_0.bin",
        "ggml-large-v3.bin",
        "ggml-base.bin",
    ];

    for dir in candidate_model_dirs() {
        for name in &model_names {
            let path = dir.join(name);
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    bail!(
        "unable to find whisper GGML model. Download one with:\n\
         curl -L -o models/ggml-large-v3-turbo.bin \\\n  \
         https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin\n\
         Or set STT_WHISPER_MODEL=/path/to/model.bin"
    )
}

fn candidate_model_dirs() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("models"));
        candidates.push(current_dir.clone());
    }

    if let Ok(executable) = env::current_exe() {
        let mut cursor = executable.parent().map(Path::to_path_buf);
        while let Some(dir) = cursor {
            candidates.push(dir.join("models"));
            candidates.push(dir.clone());
            cursor = dir.parent().map(Path::to_path_buf);
        }
    }

    candidates
}
