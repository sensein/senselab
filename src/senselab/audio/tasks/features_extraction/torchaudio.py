"""This module provides the implementation of torchaudio utilities for audio features extraction."""

from typing import Dict, List, Optional

import pydra
import torch
import torchaudio

from senselab.audio.data_structures.audio import Audio


def extract_spectrogram_from_audios(
    audios: List[Audio],
    n_fft: int = 400,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Extract spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 400.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing spectrograms.
    """
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
        spectrograms.append({"spectrogram": spectrogram(audio.waveform).squeeze(0)})
    return spectrograms


def extract_mel_spectrogram_from_audios(
    audios: List[Audio],
    n_fft: Optional[int] = 400,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
) -> List[Dict[str, torch.Tensor]]:
    """Extract mel spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 400.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        n_mels (int): Number of mel filter banks. Default is 128.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing mel spectrograms.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        if win_length is not None:
            hop_length = win_length // 2
        else:
            raise ValueError("win_length cannot be None")
    mel_spectrograms = []
    for audio in audios:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio.sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )(audio.waveform)
        mel_spectrograms.append({"mel_spectrogram": mel_spectrogram.squeeze(0)})
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
    if win_length is None:
        win_length = n_ftt
    if hop_length is None:
        if win_length is not None:
            hop_length = win_length // 2
        else:
            raise ValueError("win_length cannot be None")
    mfccs = []
    for audio in audios:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=audio.sampling_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_ftt, "win_length": win_length, "hop_length": hop_length, "n_mels": n_mels},
        )
        mfccs.append({"mfcc": mfcc_transform(audio.waveform).squeeze(0)})
    return mfccs


def extract_mel_filter_bank_from_audios(
    audios: List[Audio],
    n_mels: int = 128,
    n_stft: int = 201,
    n_fft: int = 400,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Extract mel filter bank from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mels (int): Number of mel filter banks. Default is 128.
        n_stft (int): Number of bins in STFT. Default is 201.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 400.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing mel filter banks.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2

    spectrograms = extract_spectrogram_from_audios(audios, n_fft, win_length, hop_length)

    mel_filter_banks = []
    for i, audio in enumerate(audios):
        melscale_transform = torchaudio.transforms.MelScale(
            sample_rate=audio.sampling_rate, n_mels=n_mels, n_stft=n_stft
        )
        mel_filter_banks.append({"mel_filter_bank": melscale_transform(spectrograms[i]["spectrogram"]).squeeze(0)})
    return mel_filter_banks


def extract_pitch_from_audios(
    audios: List[Audio], freq_low: int = 85, freq_high: int = 3400
) -> List[Dict[str, torch.Tensor]]:
    """Extract pitch from a list of audio objects.

    Pitch is detected using the detect_pitch_frequency function from torchaudio.
    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        audios (List[Audio]): List of Audio objects.
        freq_low (int): Lowest frequency that can be detected (Hz). Default is 85.
        freq_high (int): Highest frequency that can be detected (Hz). Default is 3400.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing pitches.
    """
    pitches = []
    for audio in audios:
        pitches.append(
            {
                "pitch": torchaudio.functional.detect_pitch_frequency(
                    audio.waveform, sample_rate=audio.sampling_rate, freq_low=freq_low, freq_high=freq_high
                ).squeeze(0)
            }
        )
    return pitches


extract_spectrogram_from_audios_pt = pydra.mark.task(extract_spectrogram_from_audios)
extract_mel_spectrogram_from_audios_pt = pydra.mark.task(extract_mel_spectrogram_from_audios)
extract_mfcc_from_audios_pt = pydra.mark.task(extract_mfcc_from_audios)
extract_mel_filter_bank_from_audios_pt = pydra.mark.task(extract_mel_filter_bank_from_audios)
extract_pitch_from_audios_pt = pydra.mark.task(extract_pitch_from_audios)
