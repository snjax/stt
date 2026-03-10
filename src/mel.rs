use rustfft::{Fft, FftPlanner, num_complex::Complex32};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_MELS: usize = 64;
pub const N_FFT: usize = 400;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
const PAD: usize = N_FFT / 2;
const MEL_FMIN: f32 = 0.0;
const MEL_FMAX: f32 = SAMPLE_RATE as f32 / 2.0;

pub struct MelSpectrogram {
    pub data: Vec<f32>,
    pub frames: usize,
}

pub fn log_mel_spectrogram(samples: &[f32]) -> MelSpectrogram {
    let mut padded = Vec::with_capacity(samples.len() + PAD * 2);
    padded.resize(PAD, 0.0);
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + PAD * 2, 0.0);

    let frames = 1 + (padded.len().saturating_sub(N_FFT)) / HOP_LENGTH;
    let window = hann_window();
    let filter_bank = mel_filter_bank();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut spectrum = vec![Complex32::default(); N_FFT];
    let mut powers = vec![0.0_f32; N_FFT / 2 + 1];
    let mut data = vec![0.0_f32; N_MELS * frames];

    for frame_index in 0..frames {
        let offset = frame_index * HOP_LENGTH;
        compute_power_spectrum(
            &padded[offset..offset + N_FFT],
            &window,
            fft.as_ref(),
            &mut spectrum,
            &mut powers,
        );

        for mel_index in 0..N_MELS {
            let mel_energy = filter_bank[mel_index]
                .iter()
                .zip(&powers)
                .map(|(weight, power)| weight * power)
                .sum::<f32>()
                .clamp(1e-9, 1e9)
                .ln();
            data[mel_index * frames + frame_index] = mel_energy;
        }
    }

    MelSpectrogram { data, frames }
}

fn compute_power_spectrum(
    frame: &[f32],
    window: &[f32],
    fft: &dyn Fft<f32>,
    spectrum: &mut [Complex32],
    output: &mut [f32],
) {
    for ((value, window_value), bin) in frame.iter().zip(window).zip(spectrum.iter_mut()) {
        *bin = Complex32::new(value * window_value, 0.0);
    }

    fft.process(spectrum);

    for (index, bin) in spectrum.iter().take(output.len()).enumerate() {
        output[index] = bin.norm_sqr();
    }
}

fn hann_window() -> Vec<f32> {
    (0..WIN_LENGTH)
        .map(|index| {
            let angle = 2.0 * std::f32::consts::PI * index as f32 / WIN_LENGTH as f32;
            0.5 - 0.5 * angle.cos()
        })
        .collect()
}

fn mel_filter_bank() -> Vec<Vec<f32>> {
    let fft_bins = N_FFT / 2 + 1;
    let all_freqs = (0..fft_bins)
        .map(|index| index as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect::<Vec<_>>();

    let mel_min = hz_to_mel(MEL_FMIN);
    let mel_max = hz_to_mel(MEL_FMAX);
    let mel_points = (0..(N_MELS + 2))
        .map(|index| {
            let ratio = index as f32 / (N_MELS + 1) as f32;
            mel_to_hz(mel_min + ratio * (mel_max - mel_min))
        })
        .collect::<Vec<_>>();

    let mut filters = vec![vec![0.0_f32; fft_bins]; N_MELS];
    for mel_index in 0..N_MELS {
        let left = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let right = mel_points[mel_index + 2];

        for (bin_index, frequency) in all_freqs.iter().enumerate() {
            let value = if *frequency < left || *frequency > right {
                0.0
            } else if *frequency <= center {
                (frequency - left) / (center - left).max(f32::EPSILON)
            } else {
                (right - frequency) / (right - center).max(f32::EPSILON)
            };
            filters[mel_index][bin_index] = value.max(0.0);
        }
    }

    filters
}

fn hz_to_mel(frequency_hz: f32) -> f32 {
    2595.0 * (1.0 + frequency_hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}
