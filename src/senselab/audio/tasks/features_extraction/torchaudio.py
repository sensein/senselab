"""This module provides the implementation of torchaudio utilities for audio features extraction."""

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

from senselab.audio.data_structures import Audio


def extract_spectrogram_from_audios(
    audios: List[Audio],
    n_fft: int = 1024,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Extract spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing spectrograms.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    spectrograms = []
    for audio in audios:
        try:
            spectrograms.append({"spectrogram": spectrogram(audio.waveform).squeeze(0)})
        except RuntimeError:
            spectrograms.append({"spectrogram": np.nan})
    return spectrograms


def extract_mel_spectrogram_from_audios(
    audios: List[Audio],
    n_fft: Optional[int] = 1024,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
) -> List[Dict[str, torch.Tensor]]:
    """Extract mel spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        n_mels (int): Number of mel filter banks. Default is 128.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing mel spectrograms.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        if win_length is not None:
            hop_length = win_length // 2
        else:
            raise ValueError("win_length cannot be None")
    mel_spectrograms = []
    for audio in audios:
        try:
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=audio.sampling_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
            )(audio.waveform)
            mel_spectrograms.append({"mel_spectrogram": mel_spectrogram.squeeze(0)})
        except RuntimeError:
            mel_spectrograms.append({"mel_spectrogram": np.nan})
    return mel_spectrograms


def extract_mfcc_from_audios(
    audios: List[Audio],
    n_mfcc: int = 40,
    n_ftt: Optional[int] = 400,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
) -> List[Dict[str, torch.Tensor]]:
    """Extract MFCCs from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mfcc (int): Number of MFCCs to return. Default is 40.
        n_ftt (int): Size of FFT, creates n_ftt // 2 + 1 bins. Default is 400.
        win_length (int): Window size. Default is None, using n_ftt.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        n_mels (int): Number of mel filter banks. Default is 128.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing MFCCs.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    if win_length is None:
        win_length = n_ftt
    if hop_length is None:
        if win_length is not None:
            hop_length = win_length // 2
        else:
            raise ValueError("win_length cannot be None")
    mfccs = []
    for audio in audios:
        try:
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=audio.sampling_rate,
                n_mfcc=n_mfcc,
                melkwargs={"n_fft": n_ftt, "win_length": win_length, "hop_length": hop_length, "n_mels": n_mels},
            )
            mfccs.append({"mfcc": mfcc_transform(audio.waveform).squeeze(0)})
        except RuntimeError:
            mfccs.append({"mfcc": np.nan})
    return mfccs


def extract_mel_filter_bank_from_audios(
    audios: List[Audio],
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Extract mel filter bank from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mels (int): Number of mel filter banks. Default is 128.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing mel filter banks.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    n_stft = n_fft // 2 + 1

    spectrograms = extract_spectrogram_from_audios(audios, n_fft, win_length, hop_length)

    mel_filter_banks = []
    for i, audio in enumerate(audios):
        try:
            melscale_transform = torchaudio.transforms.MelScale(
                sample_rate=audio.sampling_rate, n_mels=n_mels, n_stft=n_stft
            )
            mel_filter_banks.append({"mel_filter_bank": melscale_transform(spectrograms[i]["spectrogram"]).squeeze(0)})
        except RuntimeError:
            mel_filter_banks.append({"mel_filter_bank": np.nan})
    return mel_filter_banks


def extract_mel_filter_bank_from_spectrograms(
    spectrograms: List[Dict[str, torch.Tensor]],
    sampling_rate: int,
    n_mels: int = 128,
) -> List[Dict[str, torch.Tensor]]:
    """Extract mel filter bank from a list of audio objects.

    Args:
        spectrograms (List[torch.Tensor]): List of spectrograms.
        sampling_rate (int): Sampling rate of the audio.
        n_mels (int): Number of mel filter banks. Default is 128.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing mel filter banks.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    mel_filter_banks = []
    for spectrogram in spectrograms:
        try:
            melscale_transform = torchaudio.transforms.MelScale(
                sample_rate=sampling_rate, n_mels=n_mels, n_stft=spectrogram["spectrogram"].shape[0]
            )
            mel_filter_banks.append({"mel_filter_bank": melscale_transform(spectrogram["spectrogram"]).squeeze(0)})
        except RuntimeError:
            mel_filter_banks.append({"mel_filter_bank": np.nan})
    return mel_filter_banks


def extract_pitch_from_audios(
    audios: List[Audio], freq_low: int = 80, freq_high: int = 500
) -> List[Dict[str, torch.Tensor]]:
    """Extract pitch from a list of audio objects.

    Pitch is detected using the detect_pitch_frequency function from torchaudio.
    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        audios (List[Audio]): List of Audio objects.
        freq_low (int): Lowest frequency that can be detected (Hz). Should be bigger than 0.
            (Default is 80).
        freq_high (int): Highest frequency that can be detected (Hz).
            (Default is 500).

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing pitches.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. " "Please install senselab audio dependencies using `pip install senselab`."
        )

    if freq_low <= 0:
        raise ValueError("freq_low should be bigger than 0")

    pitches = []
    for audio in audios:
        try:
            pitches.append(
                {
                    "pitch": torchaudio.functional.detect_pitch_frequency(
                        audio.waveform, sample_rate=audio.sampling_rate, freq_low=freq_low, freq_high=freq_high
                    ).squeeze(0)
                }
            )
        except RuntimeError:
            pitches.append({"pitch": torch.tensor(torch.nan)})
    return pitches


def extract_torchaudio_features_from_audios(
    audios: List[Audio],
    freq_low: int = 80,
    freq_high: int = 500,
    n_fft: int = 1024,
    n_mels: int = 128,
    n_mfcc: int = 40,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract torchaudio features from a list of audio objects.

    Args:
    audios (List[Audio]): List of Audio objects.
    freq_low (int): Lowest detectable frequency (Hz). Must be > 0.
        Default is 80.
    freq_high (int): Highest detectable frequency (Hz).
        Default is 500.
    n_fft (int): Size of FFT; creates n_fft // 2 + 1 bins.
        Default is 1024.
    n_mels (int): Number of mel filter banks. Default is 128.
    n_mfcc (int): Number of MFCC coefficients. Default is 40.
    win_length (Optional[int]): Window size. If None, uses n_fft.
        Default is None.
    hop_length (Optional[int]): Hop length between STFT windows. If None, uses win_length // 2.
        Default is None.

    Returns:
    - List[Dict[str, Any]]: List of Dict objects containing features.

    Raises:
    - ModuleNotFoundError: If `torchaudio` is not installed.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    wl = n_fft if win_length is None else win_length
    hl = wl // 2 if hop_length is None else hop_length

    if freq_low <= 0:
        raise ValueError("freq_low should be bigger than 0")

    results: List[Dict[str, Any]] = []
    for sample in audios:
        sr = sample.sampling_rate
        try:
            # Spectrogram
            spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=wl, hop_length=hl)(sample.waveform)
            spec = spec.squeeze(0)  # (freq, time)

            # Mel-spectrogram
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, win_length=wl, hop_length=hl, n_mels=n_mels
            )(sample.waveform)
            melspec = melspec.squeeze(0)  # (mel, time)

            # MFCC
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=n_mfcc,
                melkwargs={"n_fft": n_fft, "win_length": wl, "hop_length": hl, "n_mels": n_mels},
            )(sample.waveform)
            mfcc = mfcc.squeeze(0)  # (n_mfcc, time)

            # Mel filter bank from spectrogram
            n_stft = n_fft // 2 + 1
            melfb = torchaudio.transforms.MelScale(sample_rate=sr, n_mels=n_mels, n_stft=n_stft)(spec)
            melfb = melfb.squeeze(0)  # (mel, time)

            # Pitch
            pitch = torchaudio.functional.detect_pitch_frequency(
                sample.waveform, sample_rate=sr, freq_low=freq_low, freq_high=freq_high
            ).squeeze(0)

            results.append(
                {
                    "pitch": pitch,
                    "mel_filter_bank": melfb,
                    "mfcc": mfcc,
                    "mel_spectrogram": melspec,
                    "spectrogram": spec,
                }
            )
        except RuntimeError:
            # Return NaNs with the expected keys if torchaudio raises
            results.append(
                {
                    "pitch": torch.tensor(torch.nan),
                    "mel_filter_bank": np.nan,
                    "mfcc": np.nan,
                    "mel_spectrogram": np.nan,
                    "spectrogram": np.nan,
                }
            )

    return results
