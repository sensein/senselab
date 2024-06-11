"""This module provides the implementation of torchaudio utilities for audio features extraction."""
from typing import Optional, List

import torch
import torchaudio

from senselab.audio.data_structures.audio import Audio


def extract_spectrogram_from_audios(audios: List[Audio], 
                                    n_fft: int = 1024, 
                                    win_length: Optional[int] = None, 
                                    hop_length: int = 512, 
                                    center: bool = True,
                                    pad_mode: str = "reflect",
                                    power: float = 2.0
                                    ) -> List[torch.Tensor]:
    """Extract spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is 512.
        center (bool): Whether to pad the input so that the t-th frame is centered at t*hop_length. Default is True.
        pad_mode (str): Padding method. Default is "reflect".
        power (float): Exponent for the magnitude spectrogram. Default is 2.0.

    Returns:
        List[torch.Tensor]: List of spectrograms.
    """
    if win_length is None:
        win_length = n_fft
    spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            power=power
        )
    spectrograms = []
    for audio in audios:
        spectrograms.append(spectrogram(audio.waveform))
    return spectrograms

def extract_mel_spectrogram_from_audios(audios: List[Audio], 
                            n_mels: int = 128, 
                            n_fft: Optional[int] = 1024, 
                            win_length: Optional[int] = None, 
                            hop_length: int = 512, 
                            center: bool = True,
                            pad_mode: str = "reflect",
                            power: float = 2.0,
                            norm: str = 'slaney',
                            onesided: bool = True,
                            mel_scale: str = 'htk'
                            ) -> List[torch.Tensor]:
    """Extract mel spectrograms from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mels (int): Number of mel filter banks. Default is 128.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is 512.
        center (bool): Whether to pad the input so that the t-th frame is centered at t*hop_length. Default is True.
        pad_mode (str): Padding method. Default is "reflect".
        power (float): Exponent for the magnitude spectrogram. Default is 2.0.
        norm (str): Normalization method. Default is 'slaney'.
        onesided (bool): Whether to return one-sided spectrogram. Default is True.
        mel_scale (str): Mel scale to use. Default is 'htk'.

    Returns:
        List[torch.Tensor]: List of mel spectrograms.
    """
    if win_length is None:
        win_length = n_fft
    mel_spectrograms = []
    for audio in audios:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio.sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            pad_mode=pad_mode,
            power=power,
            norm=norm,
            onesided=onesided,
            mel_scale=mel_scale
        )(audio.waveform)
        mel_spectrograms.append(mel_spectrogram)
    return mel_spectrograms

def extract_mfcc_from_audios(audios: List[Audio], 
                             n_ftt: int = 2048,
                             win_length: Optional[int] = None,
                             hop_length: int = 512,
                             n_mels: int = 256,
                             n_mfcc: int = 256
                             ) -> List[torch.Tensor]:
    """Extract MFCCs from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_ftt (int): Size of FFT, creates n_ftt // 2 + 1 bins. Default is 2048.
        win_length (int): Window size. Default is None, using n_ftt.
        hop_length (int): Length of hop between STFT windows. Default is 512.
        n_mels (int): Number of mel filter banks. Default is 256.
        n_mfcc (int): Number of MFCCs to return. Default is 256.

    Returns:
        List[torch.Tensor]: List of MFCCs.
    """    
    if win_length is None:
        win_length = n_ftt
    mfccs = []
    for audio in audios:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=audio.sampling_rate,
            n_mfcc=n_mfcc,
            n_ftt=n_ftt,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mfccs.append(mfcc_transform(audio.waveform))
    return mfccs

def extract_pitch_from_audios(audios: List[Audio]) -> List[torch.Tensor]:
    """Extract pitch from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.

    Returns:
        List[torch.Tensor]: List of pitches.
    """
    pitches = []
    for audio in audios:
        pitches.append(
            torchaudio.functional.detect_pitch_frequency(
            audio.waveform,
            sample_rate=audio.sampling_rate)
        )
    return pitches

'''
def extract_mel_filter_bank(audios: List[Audio],
                            n_mels: int = 64, 
                            n_fft: int = n_fft
                            ) -> List[torch.Tensor]:
    """Extract mel filter bank from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mels (int): Number of mel filter banks. Default is 64.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is n_fft.

    Returns:
        List[torch.Tensor]: List of mel filter banks.
    """
    mel_filter_banks = []

    for audio in audios:
        mel_filters = torchaudio.functional.create_fb_matrix(
            int(n_fft // 2 + 1),
            n_mels=n_mels,
            f_min=0.,
            f_max=audio.sampling_rate/2.,
            sample_rate=audio.sampling_rate,
            norm='slaney'
        )(audio.waveform)
        mel_filter_banks.append(mel_filters)
    return mel_filter_banks
'''
