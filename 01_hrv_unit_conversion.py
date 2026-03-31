ort os
import zipfile
import tempfile
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# Sampling parameters
fs = 256  # Sampling rate in Hz
dt = 1000 / fs  # Time interval in ms (≈ 3.90625 ms)
samples_per_block = 8


# Bandpass filter parameters
lowcut = 0.5   # Hz
highcut = 30.0  # Hz
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq


def bandpass_filter(data, low, high, fs, order=3):
    """Apply a Butterworth bandpass filter."""
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


def expand_timestamps(df):
    """
    Convert and expand block timestamps.
    Each timestamp (in microseconds) is converted to milliseconds and then expanded into 8 sample timestamps.
    """
    base_ts = df['steady_timestamp'].astype(float) * 0.001  # Convert to ms
    expanded_times = []
    for ts in base_ts:
        block_times = ts + np.arange(samples_per_block) * dt
        expanded_times.extend(block_times)
    return np.array(expanded_times)


def expand_signal(df, col_name):
    """
    Expand the ECG signal for each block into individual samples.
    Here, the same value is repeated for each of the 8 samples.
    """
    expanded_signal = np.repeat(df[col_name].values, samples_per_block)
    return expanded_signal


def detect_r_peaks(ecg_signal, distance=round(0.6*fs)):
    """
    Detect R-peaks in the ECG signal using SciPy's find_peaks.
    The 'distance' parameter ensures peaks are separated by at least ~0.6 seconds.
    """
    peaks, _ = find_peaks(ecg_signal, distance=distance)
    return peaks