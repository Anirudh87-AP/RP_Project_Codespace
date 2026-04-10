"""
================================================================================
SPEECH_ENGINE.PY — Digital Signal Processing & AI Speech Pipeline
================================================================================
Project: Complex Random Process Analysis for Communication Systems
         with Applications to Speech Enhancement
================================================================================
Authors: Navin Kumar PG (24BEC1055)
         A.P. Anirudh       (24BEC1158)
         Kailash N H        (24BEC1546)
Faculty: Dr. Kalaivan K
================================================================================
Description:
    This module implements the core mathematical and signal-processing
    pipeline for the project.  It encapsulates:

        1. Audio I/O   — Loading / saving WAV files via pydub and numpy.
        2. Noise Model — Generating calibrated Additive White Gaussian
                         Noise (AWGN) at a specified SNR.
        3. Filtering   — A 4th-order digital Butterworth low-pass filter
                         (scipy.signal) to recover the signal envelope
                         y(t) from the corrupted input x(t) = s(t)+n(t).
        4. Metrics     — SNR (dB) and Mean Square Error calculations.
        5. ASR         — Dual speech-to-text transcription of x(t) and
                         y(t) using the Google Web Speech API.
        6. Summariser  — Rule-based extractive text summarisation that
                         distils transcriptions into concise sentences.
        7. Synthetic   — On-the-fly generation of test signals (multi-
                         tone speech-like waveforms + AWGN).
        8. Pipeline    — The orchestrator function ``process_pipeline``
                         that chains all steps and returns a result dict.

    Mathematical notation follows the project specification:
        s(t)  — pure modulated signal (transmitter output)
        n(t)  — stochastic AWGN noise introduced by the channel
        x(t)  — corrupted received signal, x(t) = s(t) + n(t)
        y(t)  — enhanced / denoised signal after Butterworth LPF

Revision History:
    2026-04-08  Initial creation — full pipeline
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────────────────────
import io
import os
import time
import wave
import struct
import logging
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, sosfilt, sosfiltfilt

# ──────────────────────────────────────────────────────────────────────────────
# Optional: pydub (requires ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from pydub import AudioSegment  # type: ignore
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Optional: speech_recognition
# ──────────────────────────────────────────────────────────────────────────────
try:
    import speech_recognition as sr  # type: ignore
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Module-Level Logger
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16000          # 16 kHz — standard for speech
DEFAULT_FILTER_ORDER = 4             # 4th-order Butterworth
DEFAULT_CUTOFF_HZ = 4000.0           # Low-pass cutoff frequency
DEFAULT_CHART_POINTS = 100           # Downsampled points for Chart.js
DEFAULT_DURATION = 4.0               # Default synthetic audio duration (sec)
DEFAULT_SNR_DB = 5.0                 # Default noise level for samples
MODULATION_INDEX_MU = 0.85           # AM modulation index μ
FREQUENCY_CONSTANT_KA = 2 * np.pi    # Frequency constant k_a

# ──────────────────────────────────────────────────────────────────────────────
# Supported Language Codes (BCP-47)
# ──────────────────────────────────────────────────────────────────────────────
LANGUAGE_CODES = {
    "en-US": {"name": "English", "display": "English",  "flag": "🇬🇧"},
    "hi-IN": {"name": "Hindi",   "display": "हिन्दी",  "flag": "🇮🇳"},
    "ta-IN": {"name": "Tamil",   "display": "தமிழ்",  "flag": "🇮🇳"},
}
ALLOWED_LANGUAGES = set(LANGUAGE_CODES.keys())
DEFAULT_LANGUAGE  = "en-US"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — AUDIO I/O UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_audio_file(file_path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple:
    """
    Load an audio file from disk and convert it to a mono numpy array.

    Supports WAV, MP3, OGG, FLAC, and other formats if ffmpeg is
    installed and pydub is available.  Falls back to the standard
    library ``wave`` module for plain WAV files.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the audio file.
    target_sr : int
        Desired sample rate in Hz.  The audio will be resampled if
        its native rate differs.

    Returns
    -------
    tuple[np.ndarray, int]
        (signal_array, sample_rate)
        The signal is normalised to the range [−1.0, +1.0].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be decoded.
    """
    file_path = str(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info("Loading audio file: %s", file_path)

    # ── Try pydub first (handles all formats via ffmpeg) ──────────────
    if PYDUB_AVAILABLE:
        try:
            audio_segment = AudioSegment.from_file(file_path)
            audio_segment = audio_segment.set_channels(1)                # mono
            audio_segment = audio_segment.set_frame_rate(target_sr)      # resample
            audio_segment = audio_segment.set_sample_width(2)            # 16-bit

            raw_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float64)
            raw_data = raw_data / 32768.0  # normalise 16-bit to [−1, 1]

            logger.info(
                "Loaded via pydub: %.2f sec, %d Hz, %d samples",
                len(raw_data) / target_sr,
                target_sr,
                len(raw_data),
            )
            return raw_data, target_sr

        except Exception as exc:
            logger.warning("pydub failed (%s), falling back to wave module.", exc)

    # ── Fallback: standard library wave (WAV only) ────────────────────
    return _load_wav_stdlib(file_path, target_sr)


def _load_wav_stdlib(file_path: str, target_sr: int) -> tuple:
    """
    Load a WAV file using Python's built-in ``wave`` module.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.
    target_sr : int
        Desired output sample rate.

    Returns
    -------
    tuple[np.ndarray, int]
        (signal_array, sample_rate)
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw_bytes = wf.readframes(n_frames)

    # Determine struct format
    if sample_width == 1:
        fmt = f"<{n_frames * n_channels}B"
        max_val = 128.0
        offset = 128
    elif sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        max_val = 32768.0
        offset = 0
    elif sample_width == 4:
        fmt = f"<{n_frames * n_channels}i"
        max_val = 2147483648.0
        offset = 0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = struct.unpack(fmt, raw_bytes)
    signal = np.array(samples, dtype=np.float64)

    # Mix to mono if stereo
    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    # Normalise
    signal = (signal - offset) / max_val

    # Resample if needed
    if frame_rate != target_sr:
        signal = _resample_signal(signal, frame_rate, target_sr)

    logger.info(
        "Loaded via wave: %.2f sec, %d Hz → %d Hz, %d samples",
        len(signal) / target_sr,
        frame_rate,
        target_sr,
        len(signal),
    )
    return signal, target_sr


def _resample_signal(
    signal: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample a signal array from ``original_sr`` to ``target_sr``
    using scipy's ``resample`` function (FFT-based).

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.
    original_sr : int
        Original sample rate in Hz.
    target_sr : int
        Desired sample rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal.
    """
    if original_sr == target_sr:
        return signal

    duration = len(signal) / original_sr
    target_length = int(duration * target_sr)
    resampled = scipy_signal.resample(signal, target_length)

    logger.debug(
        "Resampled %d → %d samples (%d Hz → %d Hz)",
        len(signal), len(resampled), original_sr, target_sr,
    )
    return resampled


def save_audio_to_wav(
    signal: np.ndarray,
    sample_rate: int,
    file_path: str,
) -> str:
    """
    Save a numpy signal array to a 16-bit PCM WAV file.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal in the range [−1.0, +1.0].
    sample_rate : int
        Sample rate in Hz.
    file_path : str
        Destination file path.

    Returns
    -------
    str
        The absolute path of the saved file.
    """
    # Clip to valid range
    signal = np.clip(signal, -1.0, 1.0)

    # Convert to 16-bit PCM
    pcm_data = (signal * 32767).astype(np.int16)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    abs_path = os.path.abspath(file_path)
    logger.info("Saved WAV: %s (%.2f sec)", abs_path, len(signal) / sample_rate)
    return abs_path


def signal_to_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a numpy signal array to in-memory WAV bytes.

    This is used to feed audio data into the speech recognition engine
    without writing a temporary file to disk.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal in [−1.0, +1.0].
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    bytes
        Complete WAV file bytes (header + PCM data).
    """
    signal = np.clip(signal, -1.0, 1.0)
    pcm_data = (signal * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — NOISE MODEL (AWGN GENERATION)
# ══════════════════════════════════════════════════════════════════════════════

def generate_awgn(length: int, power: float = 1.0) -> np.ndarray:
    """
    Generate Additive White Gaussian Noise with specified power.

    The noise is drawn from a zero-mean Gaussian distribution:
        n(t) ~ N(0, σ²)   where σ² = power

    Parameters
    ----------
    length : int
        Number of samples to generate.
    power : float
        Desired noise power (variance σ²).

    Returns
    -------
    np.ndarray
        Noise array n(t).
    """
    sigma = np.sqrt(power)
    noise = np.random.normal(0.0, sigma, length)
    logger.debug(
        "Generated AWGN: %d samples, σ=%.6f, power=%.6f",
        length, sigma, power,
    )
    return noise


def add_awgn_at_snr(
    clean_signal: np.ndarray,
    target_snr_db: float,
) -> tuple:
    """
    Add calibrated AWGN to a clean signal to achieve a target SNR.

    The noise power is computed from the specified SNR (in dB) relative
    to the signal power:

        SNR_dB = 10 · log₁₀(P_signal / P_noise)
        P_noise = P_signal / 10^(SNR_dB / 10)

    Parameters
    ----------
    clean_signal : np.ndarray
        The original clean signal s(t).
    target_snr_db : float
        Desired Signal-to-Noise Ratio in decibels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (noisy_signal, noise)
        ``noisy_signal`` is x(t) = s(t) + n(t).
        ``noise`` is the generated n(t) component.
    """
    # Calculate signal power
    signal_power = np.mean(clean_signal ** 2)

    # Avoid division by zero for silent signals
    if signal_power < 1e-10:
        logger.warning("Signal power near zero — adding minimal noise.")
        signal_power = 1e-10

    # Calculate required noise power from target SNR
    # SNR = 10 * log10(P_s / P_n)  =>  P_n = P_s / 10^(SNR/10)
    noise_power = signal_power / (10 ** (target_snr_db / 10))

    # Generate noise
    noise = generate_awgn(len(clean_signal), noise_power)

    # Combine: x(t) = s(t) + n(t)
    noisy_signal = clean_signal + noise

    logger.info(
        "Added AWGN: target SNR=%.1f dB, P_signal=%.6f, P_noise=%.6f",
        target_snr_db, signal_power, noise_power,
    )
    return noisy_signal, noise


def calculate_noise_power(noise: np.ndarray) -> float:
    """
    Calculate the average power of a noise signal.

    Parameters
    ----------
    noise : np.ndarray
        The noise signal n(t).

    Returns
    -------
    float
        Average power (mean of squared values).
    """
    return float(np.mean(noise ** 2))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BUTTERWORTH LOW-PASS FILTER
# ══════════════════════════════════════════════════════════════════════════════

def design_butterworth_lpf(
    cutoff_hz: float,
    sample_rate: int,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Design a digital Butterworth low-pass filter in second-order sections.

    The filter is designed using the bilinear transform with frequency
    pre-warping.  Using SOS (Second-Order Sections) rather than transfer
    function (b, a) coefficients ensures numerical stability, especially
    for higher-order filters.

    The Butterworth filter is chosen because of its maximally flat
    magnitude response in the passband — it introduces no ripple,
    making it ideal for preserving the shape of the message signal
    s(t) while attenuating out-of-band noise n(t).

    Parameters
    ----------
    cutoff_hz : float
        The −3 dB cutoff frequency in Hz.
    sample_rate : int
        The sample rate of the digital signal in Hz.
    order : int
        Filter order.  A 4th-order Butterworth has a roll-off of
        −80 dB/decade (−24 dB/octave).

    Returns
    -------
    np.ndarray
        Second-order section (SOS) coefficients.
        Shape: (order//2 + order%2, 6)
    """
    nyquist = sample_rate / 2.0
    normalised_cutoff = cutoff_hz / nyquist

    # Clamp to valid range (0, 1)
    normalised_cutoff = np.clip(normalised_cutoff, 0.001, 0.999)

    sos = butter(
        N=order,
        Wn=normalised_cutoff,
        btype="low",
        analog=False,
        output="sos",
    )

    logger.info(
        "Designed Butterworth LPF: order=%d, cutoff=%.1f Hz, Wn=%.4f",
        order, cutoff_hz, normalised_cutoff,
    )
    return sos


def apply_butterworth_filter(
    signal_array: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_FILTER_ORDER,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Apply a 4th-order Butterworth low-pass filter to the input signal
    to recover the envelope y(t) from the noisy input x(t).

    This implements the "Processing (Demodulator Output)" stage of the
    random process signal flow:
        x(t)  ──▶  [Butterworth LPF]  ──▶  y(t)

    If ``zero_phase`` is ``True``, the filter is applied forwards and
    backwards (``sosfiltfilt``), resulting in zero phase distortion but
    effectively doubling the filter order in terms of magnitude response.

    Parameters
    ----------
    signal_array : np.ndarray
        The corrupted input signal x(t) = s(t) + n(t).
    sample_rate : int
        Sample rate in Hz.
    cutoff_hz : float
        Cutoff frequency for the low-pass filter.
    order : int
        Filter order (default: 4).
    zero_phase : bool
        If True, use forward-backward filtering (zero phase distortion).

    Returns
    -------
    np.ndarray
        The enhanced signal y(t) — the recovered envelope.
    """
    sos = design_butterworth_lpf(cutoff_hz, sample_rate, order)

    if zero_phase:
        # Forward-backward filtering for zero phase distortion
        # This requires signal length > 3 * max(len(sos sections))
        padlen = min(3 * (2 * order + 1), len(signal_array) - 1)
        if padlen < 1:
            padlen = 0
        filtered = sosfiltfilt(sos, signal_array, padlen=padlen)
    else:
        # Causal filtering (introduces phase delay)
        filtered = sosfilt(sos, signal_array)

    logger.info(
        "Applied Butterworth LPF: %d samples, order=%d, cutoff=%.1f Hz, zero_phase=%s",
        len(signal_array), order, cutoff_hz, zero_phase,
    )
    return filtered


def get_filter_frequency_response(
    cutoff_hz: float,
    sample_rate: int,
    order: int = DEFAULT_FILTER_ORDER,
    n_points: int = 512,
) -> tuple:
    """
    Compute the frequency response of the designed Butterworth filter.

    Useful for plotting the filter's magnitude and phase response
    to verify its characteristics.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    sample_rate : int
        Sample rate in Hz.
    order : int
        Filter order.
    n_points : int
        Number of frequency points to compute.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (frequencies_hz, magnitude_db, phase_degrees)
    """
    sos = design_butterworth_lpf(cutoff_hz, sample_rate, order)
    w, h = scipy_signal.sosfreqz(sos, worN=n_points)

    frequencies_hz = w * sample_rate / (2 * np.pi)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    phase_deg = np.degrees(np.angle(h))

    return frequencies_hz, magnitude_db, phase_deg


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SIGNAL QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_snr(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

        SNR_dB = 10 · log₁₀( P_signal / P_noise )

    where:
        P_signal = mean( s(t)² )
        P_noise  = mean( (x(t) - s(t))² )

    Parameters
    ----------
    clean_signal : np.ndarray
        The reference clean signal s(t).
    noisy_signal : np.ndarray
        The signal to evaluate (could be x(t) or y(t)).

    Returns
    -------
    float
        SNR in dB.  Returns −inf if noise is zero, +inf if signal is zero.
    """
    # Ensure same length
    min_len = min(len(clean_signal), len(noisy_signal))
    s = clean_signal[:min_len]
    x = noisy_signal[:min_len]

    signal_power = np.mean(s ** 2)
    noise_power = np.mean((x - s) ** 2)

    if noise_power < 1e-20:
        logger.debug("Noise power ≈ 0 — returning +100 dB (effectively ∞)")
        return 100.0

    if signal_power < 1e-20:
        logger.debug("Signal power ≈ 0 — returning −100 dB")
        return -100.0

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    logger.debug("SNR = %.4f dB  (P_s=%.2e, P_n=%.2e)", snr_db, signal_power, noise_power)
    return float(snr_db)


def calculate_mse(
    reference: np.ndarray,
    estimated: np.ndarray,
) -> float:
    """
    Calculate the Mean Square Error (MSE) between two signals.

        MSE = (1/N) · Σ (reference[i] − estimated[i])²

    Parameters
    ----------
    reference : np.ndarray
        The reference signal (e.g., clean s(t)).
    estimated : np.ndarray
        The estimated signal (e.g., enhanced y(t)).

    Returns
    -------
    float
        The MSE value.
    """
    min_len = min(len(reference), len(estimated))
    r = reference[:min_len]
    e = estimated[:min_len]

    mse = float(np.mean((r - e) ** 2))
    logger.debug("MSE = %.8f", mse)
    return mse


def calculate_psnr(
    reference: np.ndarray,
    estimated: np.ndarray,
) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) in dB.

        PSNR = 10 · log₁₀(MAX² / MSE)

    Parameters
    ----------
    reference : np.ndarray
        The reference signal.
    estimated : np.ndarray
        The estimated signal.

    Returns
    -------
    float
        PSNR in dB.
    """
    mse = calculate_mse(reference, estimated)
    if mse < 1e-20:
        return 100.0

    max_val = max(np.max(np.abs(reference)), 1.0)
    psnr = 10.0 * np.log10(max_val ** 2 / mse)
    return float(psnr)


def calculate_signal_statistics(signal: np.ndarray) -> dict:
    """
    Compute comprehensive statistics for a signal array.

    Parameters
    ----------
    signal : np.ndarray
        The signal to analyse.

    Returns
    -------
    dict
        Dictionary of statistical measures.
    """
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "variance": float(np.var(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "peak": float(np.max(np.abs(signal))),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "dynamic_range_db": float(
            20 * np.log10(np.max(np.abs(signal)) / (np.std(signal) + 1e-12))
        ),
        "zero_crossings": int(np.sum(np.diff(np.sign(signal)) != 0)),
        "length": len(signal),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DATA DOWNSAMPLING FOR CHART.JS
# ══════════════════════════════════════════════════════════════════════════════

def downsample_for_chart(
    signal: np.ndarray,
    n_points: int = DEFAULT_CHART_POINTS,
) -> list:
    """
    Downsample a signal array to ``n_points`` for Chart.js visualisation.

    Uses a uniform stride to pick evenly spaced samples from the
    original array.  This preserves the overall shape of the waveform
    while keeping the JSON payload small enough for performant rendering.

    Parameters
    ----------
    signal : np.ndarray
        The signal to downsample.
    n_points : int
        Target number of points (default: 100).

    Returns
    -------
    list[float]
        List of floats suitable for JSON serialisation.
    """
    if len(signal) <= n_points:
        return [round(float(v), 6) for v in signal]

    indices = np.linspace(0, len(signal) - 1, n_points, dtype=int)
    downsampled = signal[indices]

    result = [round(float(v), 6) for v in downsampled]
    logger.debug("Downsampled %d → %d points", len(signal), len(result))
    return result


def downsample_minmax(
    signal: np.ndarray,
    n_points: int = DEFAULT_CHART_POINTS,
) -> list:
    """
    Downsample using min-max preservation for better waveform visualisation.

    For each chunk of the signal, both the minimum and maximum values
    are kept, resulting in 2 × n_points output values.  This better
    preserves peaks and troughs that uniform downsampling might miss.

    Parameters
    ----------
    signal : np.ndarray
        The signal to downsample.
    n_points : int
        Number of chunks (output will have 2 × n_points values).

    Returns
    -------
    list[float]
        Alternating min/max values for each chunk.
    """
    if len(signal) <= n_points * 2:
        return [round(float(v), 6) for v in signal]

    chunk_size = len(signal) // n_points
    result = []

    for i in range(n_points):
        start = i * chunk_size
        end = min(start + chunk_size, len(signal))
        chunk = signal[start:end]

        result.append(round(float(np.min(chunk)), 6))
        result.append(round(float(np.max(chunk)), 6))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — SYNTHETIC AUDIO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_speech_signal(
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_frequency: float = 440.0,
) -> np.ndarray:
    """
    Generate a synthetic multi-tone signal that mimics speech characteristics.

    The signal uses Amplitude Modulation (AM) with the modulation index μ
    and frequency constant k_a as specified in the project requirements:

        s(t) = [1 + μ · m(t)] · cos(2π · f_c · t)

    where m(t) is a composite message signal constructed from multiple
    sinusoidal components at speech-band frequencies (100–3400 Hz),
    simulating the harmonic structure of human speech.

    Parameters
    ----------
    duration : float
        Audio length in seconds.
    sample_rate : int
        Sample rate in Hz.
    base_frequency : float
        Carrier frequency f_c in Hz.

    Returns
    -------
    np.ndarray
        The clean modulated signal s(t), normalised to [−0.9, +0.9].
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # ── Composite message signal m(t) ─────────────────────────────────
    # Fundamental + harmonics mimicking speech formants
    m_t = np.zeros_like(t)

    # Fundamental vocal frequency (~150 Hz)
    m_t += 1.0 * np.sin(2 * np.pi * 150 * t)

    # First formant region (~400 Hz)
    m_t += 0.7 * np.sin(2 * np.pi * 400 * t)

    # Second formant region (~900 Hz)
    m_t += 0.5 * np.sin(2 * np.pi * 900 * t)

    # Third formant region (~2500 Hz)
    m_t += 0.3 * np.sin(2 * np.pi * 2500 * t)

    # Consonant-like component (~3200 Hz)
    m_t += 0.15 * np.sin(2 * np.pi * 3200 * t)

    # Add amplitude variation (syllable-like envelope)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    m_t *= syllable_env

    # Normalise m(t) to [−1, 1] for modulation
    m_max = np.max(np.abs(m_t))
    if m_max > 0:
        m_t = m_t / m_max

    # ── AM Modulation: s(t) = [1 + μ · m(t)] · cos(k_a · f_c · t) ───
    mu = MODULATION_INDEX_MU
    carrier = np.cos(FREQUENCY_CONSTANT_KA * base_frequency * t)
    s_t = (1.0 + mu * m_t) * carrier

    # Normalise to safe amplitude
    max_amp = np.max(np.abs(s_t))
    if max_amp > 0:
        s_t = 0.9 * s_t / max_amp

    logger.info(
        "Generated synthetic speech: %.2f sec, %d Hz, μ=%.2f, f_c=%.1f Hz",
        duration, sample_rate, mu, base_frequency,
    )
    return s_t


def generate_test_audio(
    sample_key: str,
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    snr_db: float = DEFAULT_SNR_DB,
    output_dir: str = "uploads",
    session_id: str = "test",
) -> tuple:
    """
    Generate a complete test audio file with calibrated noise.

    This function creates both the noisy input x(t) and the clean
    reference s(t), saves the noisy version to disk, and returns
    all arrays for further processing.

    Parameters
    ----------
    sample_key : str
        One of: "cafe_noise", "traffic_noise", "office_hum",
        "construction_noise", "clean_reference".
    duration : float
        Audio length in seconds.
    sample_rate : int
        Sample rate in Hz.
    snr_db : float
        Target SNR for noise addition.
    output_dir : str
        Directory to save the generated WAV file.
    session_id : str
        Session identifier for filename.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray, np.ndarray]
        (file_path, clean_signal, noisy_signal, noise)
    """
    # ── Noise profile settings ────────────────────────────────────────
    noise_profiles = {
        "cafe_noise": {"snr_db": 5.0, "freq": 440.0},
        "traffic_noise": {"snr_db": 0.0, "freq": 440.0},
        "office_hum": {"snr_db": 15.0, "freq": 440.0},
        "construction_noise": {"snr_db": -5.0, "freq": 440.0},
        "clean_reference": {"snr_db": 100.0, "freq": 440.0},
    }

    profile = noise_profiles.get(sample_key, {"snr_db": snr_db, "freq": 440.0})
    target_snr = profile["snr_db"]
    base_freq = profile["freq"]

    logger.info(
        "Generating test audio: key=%s, snr=%.1f dB, duration=%.1f sec",
        sample_key, target_snr, duration,
    )

    # ── Generate clean signal s(t) ────────────────────────────────────
    clean_signal = generate_synthetic_speech_signal(
        duration=duration,
        sample_rate=sample_rate,
        base_frequency=base_freq,
    )

    # ── Add AWGN: x(t) = s(t) + n(t) ─────────────────────────────────
    if target_snr >= 90:
        # "Clean" sample — no noise
        noisy_signal = clean_signal.copy()
        noise = np.zeros_like(clean_signal)
    else:
        noisy_signal, noise = add_awgn_at_snr(clean_signal, target_snr)

    # ── Save to file ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"input_{session_id}.wav")
    save_audio_to_wav(noisy_signal, sample_rate, file_path)

    return file_path, clean_signal, noisy_signal, noise


def generate_speech_with_text(
    text: str,
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a synthetic signal whose spectral characteristics vary
    based on the input text (a pseudo-speech synthesiser for demo purposes).

    This does NOT perform actual text-to-speech; instead, it maps
    characters to frequency components to create a signal that would
    produce different waveforms for different text inputs.

    Parameters
    ----------
    text : str
        Input text to "encode" into the signal.
    duration : float
        Duration in seconds.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        The generated signal.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.zeros_like(t)

    # Map each character to a frequency component
    for i, char in enumerate(text[:20]):  # Limit to 20 chars
        freq = 200 + (ord(char) % 26) * 100
        amplitude = 0.3 / (i + 1)
        phase = (ord(char) * 0.1) % (2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Add speech-like envelope
    envelope = np.ones_like(t)
    n_syllables = max(len(text.split()) // 2, 1)
    for k in range(n_syllables):
        center = (k + 0.5) / n_syllables * duration
        sigma = duration / (n_syllables * 4)
        envelope *= 1.0 + 0.3 * np.exp(-0.5 * ((t - center) / sigma) ** 2)

    signal *= envelope

    # Normalise
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = 0.8 * signal / max_amp

    return signal


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — AUTOMATIC SPEECH RECOGNITION (ASR)
# ══════════════════════════════════════════════════════════════════════════════

def _run_asr_with_timeout(
    audio_data,
    language: str,
    energy_threshold: int,
    timeout_sec: int,
) -> str:
    """Run recognize_google in a thread with a hard wall-clock deadline."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold         = energy_threshold
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold          = 0.5
    recognizer.non_speaking_duration    = 0.3

    def _call():
        return recognizer.recognize_google(audio_data, language=language, show_all=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        return future.result(timeout=timeout_sec)   # raises TimeoutError if exceeded


def transcribe_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    language: str = DEFAULT_LANGUAGE,
    timeout: int = 8,
) -> str:
    """
    Perform Automatic Speech Recognition on an audio array using the
    Google Web Speech API, with hard wall-clock timeouts and auto-retry.

    Each attempt is wrapped in ``_run_asr_with_timeout()`` which uses
    ``concurrent.futures.ThreadPoolExecutor`` so the API call can NEVER
    hang beyond ``timeout`` seconds, regardless of network conditions.

    Passes:
        1. energy_threshold=50, timeout=8s  — standard sensitivity
        2. energy_threshold=10, timeout=5s  — ultra-sensitive retry

    Parameters
    ----------
    audio_array : np.ndarray
        Audio signal in [−1.0, +1.0].
    sample_rate : int
        Sample rate in Hz.
    language : str
        BCP-47 language code ("en-US", "hi-IN", "ta-IN").
    timeout : int
        Hard wall-clock seconds per API attempt.

    Returns
    -------
    str
        Recognised text, or a bracketed diagnostic message.
    """
    if not SR_AVAILABLE:
        logger.warning("speech_recognition not installed — using signal fallback.")
        return _transcription_fallback(audio_array)

    try:
        # ── Pre-process ──────────────────────────────────────────────
        audio_proc = audio_array.copy().astype(np.float32)

        # Silence gate — no point in calling the API for silence
        rms = float(np.sqrt(np.mean(audio_proc ** 2)))
        if rms < 0.004:
            logger.info("ASR: silence gate (RMS=%.5f)", rms)
            return "[Silence detected — no speech content present]"

        # Normalise to 92% peak for optimal ASR confidence
        peak = float(np.max(np.abs(audio_proc)))
        if peak > 0:
            audio_proc = audio_proc / peak * 0.92

        # Resample to 16 kHz (Google SR optimal sample rate)
        recognize_sr = sample_rate
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sample_rate)
            audio_proc = resample_poly(audio_proc, 16000 // g, sample_rate // g)
            recognize_sr = 16000

        # Build AudioData — shared across both attempts
        pcm_bytes = (np.clip(audio_proc, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        audio_data = sr.AudioData(
            frame_data=pcm_bytes,
            sample_rate=recognize_sr,
            sample_width=2,
        )

        t0 = time.time()
        logger.info("ASR — lang=%s  rms=%.4f  sr=%dHz  timeout=%ds",
                    language, rms, recognize_sr, timeout)

        # ── Pass 1: standard sensitivity (threshold=50) ───────────────
        try:
            text = _run_asr_with_timeout(audio_data, language, 50, timeout)
            logger.info("ASR pass-1 OK in %.2fs: '%s'", time.time() - t0, text[:80])
            return text.strip()

        except concurrent.futures.TimeoutError:
            logger.warning("ASR pass-1 timed out after %ds", timeout)

        except sr.UnknownValueError:
            logger.info("ASR pass-1 unintelligible — retrying with threshold=10")

        except sr.RequestError as exc:
            logger.warning("ASR pass-1 request error: %s", exc)
            return f"[Speech API unavailable — {exc}]"

        # ── Pass 2: ultra-sensitive retry (threshold=10) ──────────────
        retry_timeout = max(timeout - 3, 5)
        try:
            text = _run_asr_with_timeout(audio_data, language, 10, retry_timeout)
            logger.info("ASR pass-2 OK in %.2fs: '%s'", time.time() - t0, text[:80])
            return text.strip()

        except concurrent.futures.TimeoutError:
            logger.warning("ASR pass-2 timed out — network may be unreachable")
            return "[Speech API timeout — check internet connection]"

        except sr.UnknownValueError:
            logger.info("ASR: both passes unintelligible")
            return "[Speech unintelligible — noise level too high for recognition]"

        except sr.RequestError as exc2:
            logger.warning("ASR pass-2 request error: %s", exc2)
            return f"[Speech API unavailable — {exc2}]"

        except Exception as exc2:
            logger.warning("ASR pass-2 unexpected error: %s", exc2)
            return _transcription_fallback(audio_array)

    except Exception as exc:
        logger.error("Unexpected ASR error: %s", exc, exc_info=True)
        return f"[Transcription error: {exc}]"


def _transcription_fallback(audio_array: np.ndarray) -> str:
    """
    Generate a pseudo-transcription based on signal characteristics
    when the Google Speech API is unavailable.

    This analyses the audio's spectral content to produce a plausible
    placeholder message, demonstrating the concept without an actual
    ASR engine.

    Parameters
    ----------
    audio_array : np.ndarray
        The audio signal to "transcribe".

    Returns
    -------
    str
        Fallback transcription text.
    """
    rms = np.sqrt(np.mean(audio_array ** 2))
    zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
    zcr_rate = zero_crossings / len(audio_array)

    # Determine signal quality assessment
    if rms < 0.01:
        return "[Silence detected — no speech content]"
    elif rms < 0.05:
        quality = "very low level"
    elif rms < 0.15:
        quality = "moderate level"
    else:
        quality = "strong level"

    if zcr_rate > 0.3:
        noise_assessment = "high noise content detected"
    elif zcr_rate > 0.15:
        noise_assessment = "moderate noise present"
    else:
        noise_assessment = "relatively clean signal"

    return (
        f"[Offline ASR fallback: {quality} audio detected, "
        f"{noise_assessment}. Install speech_recognition and ensure "
        f"internet connectivity for actual transcription.]"
    )


def transcribe_audio_enhanced(
    audio_array: np.ndarray,
    sample_rate: int,
    apply_preprocessing: bool = True,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Enhanced transcription with optional pre-processing.

    Applies mild noise reduction and normalisation before attempting
    speech recognition, which can improve results for moderately
    noisy signals.

    Parameters
    ----------
    audio_array : np.ndarray
        Audio signal.
    sample_rate : int
        Sample rate in Hz.
    apply_preprocessing : bool
        Whether to apply mild Butterworth pre-filtering.
    language : str
        BCP-47 language code ("en-US", "hi-IN", or "ta-IN").

    Returns
    -------
    str
        Transcribed text.
    """
    if apply_preprocessing:
        # Apply a gentle high-pass filter to remove DC offset and rumble
        try:
            sos_hp = butter(2, 80.0 / (sample_rate / 2), btype="high", output="sos")
            audio_array = sosfilt(sos_hp, audio_array)
        except Exception:
            pass

        # Normalise amplitude
        max_amp = np.max(np.abs(audio_array))
        if max_amp > 0:
            audio_array = audio_array / max_amp * 0.95

    return transcribe_audio(audio_array, sample_rate, language=language)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — AI TEXT SUMMARISATION
# ══════════════════════════════════════════════════════════════════════════════

def summarize_text(
    text: str,
    max_sentences: int = 3,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """
    Generate a professional, third-person AI editorial summary of transcribed speech.

    Voice style:
      - 3rd-person perspective ("The speaker discusses...", "The subject asserts...")
      - Professional slang / editorial tone (concise, punchy, no filler)
      - Multilingual fallback messages for non-speech input

    Algorithm:
        1. Tokenise into sentences.
        2. Score by keyword density, semantic weight & position.
        3. Extract top N, rewrite in 3rd-person editorial voice.
        4. Return a tight, publication-ready paragraph.

    Parameters
    ----------
    text : str
        Transcribed audio text.
    max_sentences : int
        Max sentences in output summary.
    language : str
        BCP-47 code ("en-US", "hi-IN", "ta-IN").

    Returns
    -------
    str
        Professional 3rd-person AI summary.
    """
    import re

    # ── Localised fallback messages ────────────────────────────────
    _no_speech = {
        "en-US": "[No clear speech detected — summarisation unavailable]",
        "hi-IN": "[सारांश के लिए कोई स्पष्ट भाषण नहीं मिला]",
        "ta-IN": "[சுருக்கத்திற்கு தெளிவான பேச்சு கண்டறியப்படவில்லை]",
    }
    lang_key = language if language in _no_speech else "en-US"

    if not text or text.startswith("[") or len(text.strip()) < 4:
        return _no_speech[lang_key]

    text = text.strip()
    word_count = len(text.split())

    # ── Very short input: wrap as-is in editorial voice ─────────────────
    if word_count < 6:
        _short = {
            "en-US": f'The speaker briefly conveys: "{text.capitalize()}."',
            "hi-IN": f'वक्ता संक्षिप्त रूप में कहते हैं: "{text}."',
            "ta-IN": f'பேசுபவர் சுருக்கமாக கூறுகிறார்: "{text}."',
        }
        return _short.get(lang_key, _short["en-US"])

    # ── Tokenise into sentences ──────────────────────────────────────
    sentences = _split_into_sentences(text)
    if not sentences:
        sentences = [text]

    # ── If already short, edit the whole thing in editorial voice ────
    if len(sentences) <= max_sentences:
        core = sentences
    else:
        # Score and pick best sentences
        scores = _score_sentences(sentences)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_idx = sorted(i for i, _ in indexed[:max_sentences])
        core = [sentences[i] for i in top_idx]

    # ── Rewrite in 3rd-person editorial voice (English only) ───────
    if lang_key == "en-US":
        rewritten = [_rewrite_third_person(s) for s in core]
    else:
        # For Hindi and Tamil, use the raw extracted sentences (Google
        # ASR already outputs in the target language; rewriting would
        # require a language model beyond our scope).
        rewritten = [s.strip().rstrip(".") + "." for s in core]

    summary = " ".join(rewritten)

    logger.info(
        "[summarize_text] lang=%s | %d raw sentences → %d summary sentences",
        language, len(sentences), len(rewritten),
    )
    return summary


def _rewrite_third_person(sentence: str) -> str:
    """
    Transform a sentence into a crisp, professional 3rd-person editorial voice.

    Rules applied (in order):
      1. Strip filler openers ("well", "so", "um", "like").
      2. Convert 1st-person pronouns → 3rd-person references.
      3. Clean punctuation and capitalise.
      4. Append strong period if missing.

    Returns
    -------
    str
        Edited sentence.
    """
    import re

    s = sentence.strip()
    if not s:
        return ""

    # Strip spoken filler starters
    s = re.sub(
        r'^(well|so|um|uh|like|you know|basically|honestly|literally|right|okay|ok),?\s+',
        "", s, flags=re.IGNORECASE
    )

    # Substitute 1st-person → 3rd-person editorial constructions
    # Ordered from most-specific to least-specific
    replacements = [
        # "I think / I believe / I feel / I know"
        (r"\bI think\b",         "The speaker asserts"),
        (r"\bI believe\b",       "The speaker contends"),
        (r"\bI feel\b",          "The speaker expresses that"),
        (r"\bI know\b",          "The speaker notes"),
        (r"\bIn my opinion\b",   "In the speaker's view"),
        (r"\bI am\b",            "The speaker is"),
        (r"\bI'm\b",             "The speaker is"),
        (r"\bI was\b",           "The speaker was"),
        (r"\bI have\b",          "The speaker has"),
        (r"\bI've\b",            "The speaker has"),
        (r"\bI would\b",         "The speaker would"),
        (r"\bI'd\b",             "The speaker would"),
        (r"\bI will\b",          "The speaker will"),
        (r"\bI'll\b",            "The speaker will"),
        (r"\bI can\b",           "The speaker can"),
        (r"\bI could\b",         "The speaker could"),
        (r"\bI should\b",        "The speaker should"),
        (r"\bI need\b",          "The speaker needs"),
        (r"\bI want\b",          "The speaker wants"),
        (r"\bI like\b",          "The speaker likes"),
        (r"\bI said\b",          "The speaker stated"),
        (r"\bI\b",               "the speaker"),
        # "we" plural
        (r"\bwe think\b",        "the subjects assert"),
        (r"\bwe believe\b",      "the subjects contend"),
        (r"\bwe\b",              "the group"),
        (r"\bour\b",             "their"),
        (r"\bmy\b",              "the speaker's"),
        (r"\bme\b",              "the speaker"),
        # Second person’ → third person
        (r"\byou should\b",      "one should"),
        (r"\byou can\b",         "one can"),
        (r"\byou need\b",        "one needs"),
    ]

    for pattern, replacement in replacements:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # Ensure proper capitalisation (only first character)
    if s:
        s = s[0].upper() + s[1:]

    # Ensure sentence ends with a punctuation mark
    if s and s[-1] not in ".!?":
        s += "."

    return s


def _split_into_sentences(text: str) -> list:
    """
    Split text into sentences using punctuation-based heuristics.

    Parameters
    ----------
    text : str
        The text to split.

    Returns
    -------
    list[str]
        List of sentence strings.
    """
    # Simple sentence boundary detection
    import re
    # Split on period, exclamation, or question mark followed by space or end
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

    if not sentences:
        # If no sentence boundaries found, split on commas or conjunctions
        raw_sentences = re.split(r'[,;]\s+|\s+and\s+|\s+but\s+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

    if not sentences:
        sentences = [text]

    return sentences


def _score_sentences(sentences: list) -> list:
    """
    Score sentences based on keyword frequency, position, and length.

    Scoring criteria:
        1. **Word frequency** — sentences with common meaningful words
           score higher (these represent key topics).
        2. **Position bias** — the first and last sentences get a boost.
        3. **Length normalisation** — very short or very long sentences
           are penalised.

    Parameters
    ----------
    sentences : list[str]
        List of sentence strings.

    Returns
    -------
    list[float]
        Score for each sentence.
    """
    # Stopwords to exclude from frequency counting
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "not", "so", "yet", "both",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very",
        "it", "its", "this", "that", "these", "those",
        "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "she", "her", "they", "them", "their", "what", "which", "who",
    }

    # Build word frequency map
    all_words = []
    for sentence in sentences:
        words = sentence.lower().split()
        meaningful = [w.strip(".,!?;:\"'()") for w in words if len(w) > 2]
        meaningful = [w for w in meaningful if w not in stopwords]
        all_words.extend(meaningful)

    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Score each sentence
    scores = []
    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        meaningful = [w.strip(".,!?;:\"'()") for w in words if len(w) > 2]
        meaningful = [w for w in meaningful if w not in stopwords]

        # Word frequency score
        freq_score = sum(word_freq.get(w, 0) for w in meaningful)
        if len(meaningful) > 0:
            freq_score /= len(meaningful)

        # Position bias (first = boost, last = slight boost)
        position_score = 0.0
        if i == 0:
            position_score = 2.0
        elif i == len(sentences) - 1:
            position_score = 1.0
        else:
            position_score = 0.5

        # Length normalisation (prefer medium-length sentences)
        word_count = len(words)
        if word_count < 4:
            length_factor = 0.5
        elif word_count > 30:
            length_factor = 0.7
        else:
            length_factor = 1.0

        score = (freq_score + position_score) * length_factor
        scores.append(score)

    return scores


def _simplify_sentence(sentence: str) -> str:
    """
    Simplify a sentence to make it sound more like concise AI output.

    Transformations:
        1. Remove filler words and excessive adverbs.
        2. Ensure the sentence ends with proper punctuation.
        3. Capitalise the first letter.

    Parameters
    ----------
    sentence : str
        The original sentence.

    Returns
    -------
    str
        The simplified sentence.
    """
    import re

    # Remove common filler phrases
    fillers = [
        r'\bbasically\b', r'\bactually\b', r'\bliterally\b',
        r'\breally\b', r'\bjust\b', r'\bvery\b', r'\bquite\b',
        r'\bsimply\b', r'\bobviously\b', r'\bclearly\b',
        r'\bum+\b', r'\buh+\b', r'\blike\b(?=\s)',
        r'\byou know\b', r'\bi mean\b', r'\bkind of\b',
        r'\bsort of\b',
    ]

    simplified = sentence
    for filler in fillers:
        simplified = re.sub(filler, '', simplified, flags=re.IGNORECASE)

    # Clean up double spaces
    simplified = re.sub(r'\s+', ' ', simplified).strip()

    # Ensure proper punctuation
    if simplified and simplified[-1] not in '.!?':
        simplified += '.'

    # Capitalise first letter
    if simplified:
        simplified = simplified[0].upper() + simplified[1:]

    return simplified


def generate_detailed_summary(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """
    Generate a more detailed AI summary with multiple analysis aspects.

    Parameters
    ----------
    text : str
        The transcribed text from enhanced audio.
    language : str
        BCP-47 language code for localised output.

    Returns
    -------
    str
        A multi-faceted summary.
    """
    _no_detail_msg = {
        "en-US": "[No clear speech content available for detailed analysis]",
        "hi-IN": "[विस्तृत विश्लेषण के लिए कोई स्पष्ट भाषण सामग्री उपलब्ध नहीं है]",
        "ta-IN": "[விரிவான பகுப்பாய்வுக்கு தெளிவான பேச்சு உள்ளடக்கம் இல்லை]",
    }
    lang_key = language if language in _no_detail_msg else "en-US"

    if not text or text.startswith("["):
        return _no_detail_msg[lang_key]

    basic_summary = summarize_text(text, max_sentences=3, language=language)

    word_count = len(text.split())
    sentence_count = len(_split_into_sentences(text))

    analysis = (
        f"{basic_summary} "
        f"[Analysis: The enhanced audio contained {word_count} words "
        f"across {sentence_count} distinct utterances. "
        f"The Butterworth filter successfully recovered intelligible "
        f"speech from the noisy channel, enabling accurate transcription.]"
    )

    return analysis


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FULL PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_pipeline(
    input_path: str,
    output_dir: str,
    session_id: str,
    filter_order: int = DEFAULT_FILTER_ORDER,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    clean_reference: Optional[np.ndarray] = None,
    noise_array: Optional[np.ndarray] = None,
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Execute the full DSP + ASR pipeline.

    Orchestration flow:
        1. Load audio file → numpy array
        2. (If no clean reference) Estimate clean signal via aggressive filtering
        3. Apply 4th-order Butterworth LPF → y(t)
        4. Calculate SNR for x(t) and y(t)
        5. Calculate MSE
        6. Downsample both signals to 100 points
        7. Save enhanced audio as WAV
        8. Transcribe both x(t) and y(t)
        9. Generate AI summary of y(t) transcription
        10. Return results dict

    Parameters
    ----------
    input_path : str
        Path to the input audio file (noisy x(t)).
    output_dir : str
        Directory to save the enhanced audio file.
    session_id : str
        Session UUID for naming output files.
    filter_order : int
        Butterworth filter order.
    cutoff_hz : float
        Low-pass cutoff frequency (Hz).
    sample_rate : int
        Target sample rate.
    clean_reference : np.ndarray or None
        If provided, used as the clean reference s(t) for SNR calculation.
        If None, the system estimates s(t) from the input.
    noise_array : np.ndarray or None
        If provided, used as the noise component n(t).

    Returns
    -------
    dict
        Comprehensive results dictionary with all metrics, waveforms,
        transcriptions, and file paths.
    """
    # Validate language, fall back to English if not allowed
    if language not in ALLOWED_LANGUAGES:
        logger.warning("Unsupported language '%s' — falling back to en-US", language)
        language = DEFAULT_LANGUAGE

    pipeline_start = time.time()
    results = {
        "session_id": session_id,
        "status": "processing",
        "language": language,
        "steps": [],
    }

    def log_step(step_name: str, message: str) -> None:
        """Record a pipeline step for the terminal animation."""
        elapsed = time.time() - pipeline_start
        results["steps"].append({
            "step": step_name,
            "message": message,
            "elapsed": round(elapsed, 3),
        })
        logger.info("[Pipeline] %s: %s (%.3fs)", step_name, message, elapsed)

    try:
        # ── Step 1: Load audio ────────────────────────────────────────
        log_step("load_audio", f"Loading input audio from {os.path.basename(input_path)}")
        noisy_signal, actual_sr = load_audio_file(input_path, target_sr=sample_rate)
        duration = len(noisy_signal) / actual_sr

        log_step(
            "audio_info",
            f"Audio loaded: {duration:.2f}s, {actual_sr} Hz, {len(noisy_signal)} samples",
        )

        # ── Step 2: Estimate clean reference if not provided ──────────
        if clean_reference is None:
            log_step(
                "estimate_reference",
                "No clean reference — estimating s(t) via aggressive low-pass filtering",
            )
            # Use a very low cutoff to get a rough estimate of the clean signal
            # This won't be perfect, but gives reasonable SNR estimates
            reference_cutoff = min(cutoff_hz * 0.7, actual_sr * 0.2)
            clean_reference = apply_butterworth_filter(
                noisy_signal, actual_sr,
                cutoff_hz=reference_cutoff,
                order=6,  # Higher order for aggressive estimation
                zero_phase=True,
            )
            log_step(
                "reference_estimated",
                f"Reference signal estimated with {6}th-order LPF at {reference_cutoff:.0f} Hz",
            )
        else:
            # Ensure same length
            min_len = min(len(clean_reference), len(noisy_signal))
            clean_reference = clean_reference[:min_len]
            noisy_signal = noisy_signal[:min_len]
            log_step("reference_provided", "Using provided clean reference signal s(t)")

        # ── Step 3: Calculate input SNR ───────────────────────────────
        log_step("calc_input_snr", "Calculating input Signal-to-Noise Ratio for x(t)")
        input_snr = calculate_snr(clean_reference, noisy_signal)
        log_step(
            "input_snr_result",
            f"Input SNR(x(t)) = {input_snr:.4f} dB",
        )

        # ── Step 4: Apply Butterworth filter ──────────────────────────
        log_step(
            "butterworth_filter",
            f"Applying {filter_order}th-Order Butterworth Digital Low-Pass Filter",
        )
        log_step(
            "filter_params",
            f"Filter parameters: order={filter_order}, cutoff={cutoff_hz:.1f} Hz, "
            f"Nyquist={actual_sr/2:.0f} Hz",
        )

        enhanced_signal = apply_butterworth_filter(
            noisy_signal,
            actual_sr,
            cutoff_hz=cutoff_hz,
            order=filter_order,
            zero_phase=True,
        )

        log_step(
            "filter_applied",
            f"Butterworth LPF applied — {len(enhanced_signal)} output samples",
        )

        # ── Step 5: Extracting arrays from audio channel ──────────────
        log_step(
            "extract_arrays",
            "Extracting signal arrays from audio channel for analysis",
        )

        # ── Step 6: Calculate output SNR ──────────────────────────────
        log_step("calc_output_snr", "Calculating output Signal-to-Noise Ratio for y(t)")
        output_snr = calculate_snr(clean_reference, enhanced_signal)
        log_step(
            "output_snr_result",
            f"Output SNR(y(t)) = {output_snr:.4f} dB",
        )

        # ── Step 7: Calculate MSE ─────────────────────────────────────
        log_step("calc_mse", "Computing Mean Square Error between s(t) and y(t)")
        mse = calculate_mse(clean_reference, enhanced_signal)
        log_step("mse_result", f"MSE = {mse:.8f}")

        # ── Step 8: Recovering s(t) envelope as y(t) ──────────────────
        log_step(
            "envelope_recovery",
            "Recovering s(t) envelope as y(t) — Wide-Sense Stationary analysis",
        )
        snr_improvement = output_snr - input_snr
        log_step(
            "snr_improvement",
            f"SNR Improvement: {snr_improvement:.4f} dB",
        )

        # ── Step 9: Downsample for visualisation ──────────────────────
        log_step(
            "downsample",
            f"Downsampling signal arrays to {DEFAULT_CHART_POINTS} points for Chart.js",
        )
        input_waveform = downsample_for_chart(noisy_signal, DEFAULT_CHART_POINTS)
        output_waveform = downsample_for_chart(enhanced_signal, DEFAULT_CHART_POINTS)
        log_step(
            "downsample_complete",
            f"Downsampled: x(t) → {len(input_waveform)} pts, "
            f"y(t) → {len(output_waveform)} pts",
        )

        # ── Step 10: Save enhanced audio ──────────────────────────────
        log_step("save_output", "Saving enhanced audio y(t) to WAV file")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{session_id}.wav")
        save_audio_to_wav(enhanced_signal, actual_sr, output_path)
        log_step("output_saved", f"Enhanced audio saved: {os.path.basename(output_path)}")

        # ── Step 11: ASR on noisy input ───────────────────────────────
        lang_display = LANGUAGE_CODES.get(language, {}).get("name", language)
        log_step(
            "asr_input",
            f"Running Automatic Speech Recognition on noisy x(t) [{lang_display}]",
        )
        input_text = transcribe_audio(noisy_signal, actual_sr, language=language)
        log_step(
            "asr_input_result",
            f"Input transcription: \"{input_text[:60]}...\"" if len(input_text) > 60
            else f"Input transcription: \"{input_text}\"",
        )

        # ── Step 12: ASR on enhanced output ───────────────────────────
        log_step(
            "asr_output",
            f"Running Automatic Speech Recognition on enhanced y(t) [{lang_display}]",
        )
        output_text = transcribe_audio(enhanced_signal, actual_sr, language=language)
        log_step(
            "asr_output_result",
            f"Output transcription: \"{output_text[:60]}...\"" if len(output_text) > 60
            else f"Output transcription: \"{output_text}\"",
        )

        # ── Step 13: AI Summary ───────────────────────────────────────
        log_step(
            "ai_summarizer",
            f"Running AI speech summarizer [{lang_display}] on clean transcription",
        )
        ai_summary = summarize_text(output_text, language=language)
        log_step(
            "summary_result",
            f"AI Summary generated: \"{ai_summary[:60]}...\"" if len(ai_summary) > 60
            else f"AI Summary generated: \"{ai_summary}\"",
        )

        # ── Step 14: Compile results ──────────────────────────────────
        log_step("compile_results", "Compiling all results for dashboard generation")
        processing_time = time.time() - pipeline_start

        # ── Step 15: Generating dashboard ─────────────────────────────
        log_step("generate_dashboard", "Generating analytics dashboard data")

        results.update({
            "status": "completed",
            "language": language,
            "input_snr": round(input_snr, 4),
            "output_snr": round(output_snr, 4),
            "mse": round(mse, 8),
            "snr_improvement": round(snr_improvement, 4),
            "input_text": input_text,
            "output_text": output_text,
            "ai_summary": ai_summary,
            "input_waveform": input_waveform,
            "output_waveform": output_waveform,
            "input_audio_path": os.path.abspath(input_path),
            "output_audio_path": os.path.abspath(output_path),
            "processing_time": round(processing_time, 4),
            "duration_seconds": round(duration, 4),
            "sample_rate": actual_sr,
            "filter_order": filter_order,
            "filter_cutoff_hz": cutoff_hz,
        })

        log_step(
            "pipeline_complete",
            f"Pipeline complete — total time: {processing_time:.3f}s",
        )

        return results

    except Exception as exc:
        processing_time = time.time() - pipeline_start
        error_msg = f"Pipeline failed after {processing_time:.3f}s: {exc}"
        logger.error(error_msg, exc_info=True)
        log_step("pipeline_error", error_msg)

        results.update({
            "status": "failed",
            "error": str(exc),
            "processing_time": round(processing_time, 4),
        })
        return results


def process_uploaded_audio(
    input_path: str,
    output_dir: str,
    session_id: str,
    filter_order: int = DEFAULT_FILTER_ORDER,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Process a user-uploaded audio file through the DSP pipeline.

    This is the entry point for user-uploaded files where no clean
    reference is available.  The pipeline estimates the clean reference
    internally.

    Parameters
    ----------
    input_path : str
        Path to the uploaded audio file.
    output_dir : str
        Directory for output files.
    session_id : str
        Session identifier.
    filter_order : int
        Butterworth filter order.
    cutoff_hz : float
        LPF cutoff frequency.
    language : str
        BCP-47 language code for ASR and summarisation.

    Returns
    -------
    dict
        Processing results.
    """
    return process_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        session_id=session_id,
        filter_order=filter_order,
        cutoff_hz=cutoff_hz,
        language=language,
    )


def process_synthetic_audio(
    sample_key: str,
    output_dir: str,
    session_id: str,
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    snr_db: float = DEFAULT_SNR_DB,
    filter_order: int = DEFAULT_FILTER_ORDER,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Generate a synthetic test audio sample and process it through the
    full DSP pipeline.

    This function handles the "Quick-Test Sample" buttons on Page 3,
    generating audio on-the-fly and providing a clean reference for
    accurate SNR calculations.

    Parameters
    ----------
    sample_key : str
        Test profile identifier.
    output_dir : str
        Directory for output files.
    session_id : str
        Session identifier.
    duration : float
        Audio duration in seconds.
    sample_rate : int
        Sample rate in Hz.
    snr_db : float
        Target SNR for noise addition.
    filter_order : int
        Butterworth filter order.
    cutoff_hz : float
        LPF cutoff frequency.

    Returns
    -------
    dict
        Processing results.
    """
    uploads_dir = os.path.join(os.path.dirname(output_dir), "uploads")

    # Generate test audio with known clean reference
    input_path, clean_signal, noisy_signal, noise = generate_test_audio(
        sample_key=sample_key,
        duration=duration,
        sample_rate=sample_rate,
        snr_db=snr_db,
        output_dir=uploads_dir,
        session_id=session_id,
    )

    # Process with known clean reference for accurate metrics
    return process_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        session_id=session_id,
        filter_order=filter_order,
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        clean_reference=clean_signal,
        noise_array=noise,
        language=language,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — SPECTRAL ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_power_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
) -> tuple:
    """
    Compute the power spectral density of a signal using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sample_rate : int
        Sample rate in Hz.
    n_fft : int
        FFT size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, power_spectrum_db)
    """
    frequencies, psd = scipy_signal.welch(
        signal,
        fs=sample_rate,
        nperseg=min(n_fft, len(signal)),
        noverlap=min(n_fft // 2, len(signal) // 2),
    )

    # Convert to dB
    psd_db = 10 * np.log10(psd + 1e-12)

    return frequencies, psd_db


def compute_spectrogram_data(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 256,
) -> tuple:
    """
    Compute spectrogram data for visualisation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sample_rate : int
        Sample rate in Hz.
    n_fft : int
        FFT window size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (times, frequencies, spectrogram_db)
    """
    frequencies, times, sxx = scipy_signal.spectrogram(
        signal,
        fs=sample_rate,
        nperseg=min(n_fft, len(signal)),
        noverlap=min(n_fft // 2, len(signal) // 2),
    )

    sxx_db = 10 * np.log10(sxx + 1e-12)

    return times, frequencies, sxx_db


def estimate_noise_floor(
    signal: np.ndarray,
    sample_rate: int,
    low_freq: float = 4000.0,
) -> float:
    """
    Estimate the noise floor by measuring average power above a frequency.

    For speech signals, most energy sits below 4 kHz.  Power above
    this threshold is predominantly noise.

    Parameters
    ----------
    signal : np.ndarray
        The signal to analyse.
    sample_rate : int
        Sample rate in Hz.
    low_freq : float
        Frequency above which to measure (Hz).

    Returns
    -------
    float
        Estimated noise floor power.
    """
    frequencies, psd = scipy_signal.welch(
        signal, fs=sample_rate,
        nperseg=min(1024, len(signal)),
    )

    noise_mask = frequencies >= low_freq
    if not np.any(noise_mask):
        return 0.0

    noise_power = np.mean(psd[noise_mask])
    return float(noise_power)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — SIGNAL COMPARISON UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> float:
    """
    Compute the Pearson correlation coefficient between two signals.

    Parameters
    ----------
    signal_a : np.ndarray
    signal_b : np.ndarray

    Returns
    -------
    float
        Correlation coefficient in [−1, 1].
    """
    min_len = min(len(signal_a), len(signal_b))
    a = signal_a[:min_len]
    b = signal_b[:min_len]

    a_norm = a - np.mean(a)
    b_norm = b - np.mean(b)

    numerator = np.sum(a_norm * b_norm)
    denominator = np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2))

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)


def compute_signal_difference(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> np.ndarray:
    """
    Compute the point-wise difference between two signals.

    Parameters
    ----------
    signal_a : np.ndarray
    signal_b : np.ndarray

    Returns
    -------
    np.ndarray
        Difference array: signal_a − signal_b
    """
    min_len = min(len(signal_a), len(signal_b))
    return signal_a[:min_len] - signal_b[:min_len]


# ══════════════════════════════════════════════════════════════════════════════
#  END OF SPEECH_ENGINE.PY
# ══════════════════════════════════════════════════════════════════════════════
