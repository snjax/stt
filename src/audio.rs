use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Default)]
pub struct AudioRecorder {
    buffer: Arc<Mutex<Vec<f32>>>,
    stream: Option<Stream>,
    input_sample_rate: u32,
}

impl AudioRecorder {
    pub fn start(&mut self) -> Result<()> {
        if self.stream.is_some() {
            bail!("recording already in progress");
        }

        self.buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("audio buffer lock poisoned"))?
            .clear();

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input audio device available")?;
        let supported = choose_input_config(&device)?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        self.input_sample_rate = config.sample_rate.0;

        let channels = usize::from(config.channels);
        let stream = match sample_format {
            SampleFormat::F32 => {
                build_input_stream::<f32>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::F64 => {
                build_input_stream::<f64>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::I8 => {
                build_input_stream::<i8>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::I16 => {
                build_input_stream::<i16>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::I32 => {
                build_input_stream::<i32>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::I64 => {
                build_input_stream::<i64>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::U8 => {
                build_input_stream::<u8>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::U16 => {
                build_input_stream::<u16>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::U32 => {
                build_input_stream::<u32>(&device, &config, channels, self.buffer.clone())?
            }
            SampleFormat::U64 => {
                build_input_stream::<u64>(&device, &config, channels, self.buffer.clone())?
            }
            other => bail!("unsupported input sample format: {other:?}"),
        };

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop(&mut self) -> Result<Vec<f32>> {
        if self.stream.take().is_none() {
            bail!("recording is not active");
        }

        let captured = self
            .buffer
            .lock()
            .map_err(|_| anyhow::anyhow!("audio buffer lock poisoned"))?
            .clone();

        if self.input_sample_rate == TARGET_SAMPLE_RATE {
            Ok(captured)
        } else {
            Ok(resample_linear(
                &captured,
                self.input_sample_rate,
                TARGET_SAMPLE_RATE,
            ))
        }
    }
}

pub fn load_wav_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav file {}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));

    let mono = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples = reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read floating point samples from wav")?;
            downmix_to_mono(&samples, channels)
        }
        hound::SampleFormat::Int => {
            let scale =
                ((1_i64 << (spec.bits_per_sample.saturating_sub(1) as u32)) - 1).max(1) as f32;
            let samples = reader
                .samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read integer samples from wav")?;
            let normalized = samples
                .into_iter()
                .map(|sample| (sample as f32 / scale).clamp(-1.0, 1.0))
                .collect::<Vec<_>>();
            downmix_to_mono(&normalized, channels)
        }
    };

    if spec.sample_rate == TARGET_SAMPLE_RATE {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE))
    }
}

pub fn resample_linear(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input.is_empty() || input_rate == output_rate {
        return input.to_vec();
    }
    if input.len() == 1 {
        return vec![input[0]];
    }

    let output_len =
        ((input.len() as f64 * output_rate as f64) / input_rate as f64).round() as usize;
    let output_len = output_len.max(1);
    let ratio = input_rate as f64 / output_rate as f64;

    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let position = index as f64 * ratio;
        let left = position.floor() as usize;
        let right = (left + 1).min(input.len() - 1);
        let frac = (position - left as f64) as f32;
        let sample = input[left] * (1.0 - frac) + input[right] * frac;
        output.push(sample);
    }
    output
}

fn choose_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    if let Ok(configs) = device.supported_input_configs() {
        let preferred = configs
            .filter(|config| {
                config.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                    && config.max_sample_rate().0 >= TARGET_SAMPLE_RATE
            })
            .min_by_key(|config| (config.channels() != 1, format_rank(config.sample_format())))
            .map(|config| config.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));

        if let Some(config) = preferred {
            return Ok(config);
        }
    }

    device
        .default_input_config()
        .context("failed to query default input configuration")
}

fn format_rank(format: SampleFormat) -> usize {
    match format {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::U16 => 2,
        _ => 3,
    }
}

fn downmix_to_mono(input: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return input.to_vec();
    }

    input
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect()
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: usize,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<Stream>
where
    T: Sample + SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let error_callback = |error| eprintln!("audio stream error: {error}");
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut local = Vec::with_capacity(data.len().div_ceil(channels.max(1)));
            for frame in data.chunks(channels.max(1)) {
                let sample = frame
                    .iter()
                    .map(|sample| sample.to_sample::<f32>())
                    .sum::<f32>()
                    / frame.len().max(1) as f32;
                local.push(sample.clamp(-1.0, 1.0));
            }

            if let Ok(mut shared) = buffer.lock() {
                shared.extend_from_slice(&local);
            }
        },
        error_callback,
        None,
    )?;
    Ok(stream)
}
