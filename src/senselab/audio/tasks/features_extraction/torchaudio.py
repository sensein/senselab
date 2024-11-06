"""This module provides the implementation of torchaudio utilities for audio features extraction."""

from typing import Any, Dict, List, Optional

import pydra
import torch
import torchaudio

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
            spectrograms.append({"spectrogram": torch.nan})
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
            mel_spectrograms.append({"mel_spectrogram": torch.nan})
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
        try:
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=audio.sampling_rate,
                n_mfcc=n_mfcc,
                melkwargs={"n_fft": n_ftt, "win_length": win_length, "hop_length": hop_length, "n_mels": n_mels},
            )
            mfccs.append({"mfcc": mfcc_transform(audio.waveform).squeeze(0)})
        except RuntimeError:
            mfccs.append({"mfcc": torch.nan})
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
            mel_filter_banks.append({"mel_filter_bank": torch.nan})
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
            pitches.append({"pitch": torch.nan})
    return pitches


def extract_torchaudio_features_from_audios(audios: List[Audio], 
                                            freq_low: int = 80,
                                            freq_high: int = 500,
                                            n_fft: int = 1024,
                                            n_mels: int = 128,
                                            n_mfcc: int = 40,
                                            win_length: Optional[int] = None,
                                            hop_length: Optional[int] = None,
                                            plugin: str = "cf") -> List[Dict[str, Any]]:
    """Extract torchaudio features from a list of audio objects.

    Args:
        audios (List[Audio]): The list of audio objects to extract features from.
        freq_low (int): Lowest frequency that can be detected (Hz). Should be bigger than 0.
            (Default is 80).
        freq_high (int): Highest frequency that can be detected (Hz).
            (Default is 500).
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        n_mels (int): Number of mel filter banks. Default is 128.
        n_mfcc (int): Number of MFCCs. Default is 40.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        plugin (str): The plugin to use. Default is "cf".

    Returns:
        List[Dict[str, Any]]: The list of feature dictionaries for each audio.
    """
    extract_pitch_from_audios_pt = pydra.mark.task(extract_pitch_from_audios)
    extract_mel_filter_bank_from_audios_pt = pydra.mark.task(extract_mel_filter_bank_from_audios)
    extract_mfcc_from_audios_pt = pydra.mark.task(extract_mfcc_from_audios)
    extract_mel_spectrogram_from_audios_pt = pydra.mark.task(extract_mel_spectrogram_from_audios)
    extract_spectrogram_from_audios_pt = pydra.mark.task(extract_spectrogram_from_audios)

    formatted_audios = [[audio] for audio in audios]
    wf = pydra.Workflow(name="wf", input_spec=["x"])
    wf.split("x", x=formatted_audios)
    wf.add(extract_pitch_from_audios_pt(name="extract_pitch_from_audios_pt", 
                                        audios=wf.lzin.x,
                                        freq_low=freq_low, 
                                        freq_high=freq_high))
    wf.add(extract_mel_filter_bank_from_audios_pt(name="extract_mel_filter_bank_from_audios_pt", \
                                                  audios=wf.lzin.x,
                                                  n_mels=n_mels,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length))
    wf.add(extract_mfcc_from_audios_pt(name="extract_mfcc_from_audios_pt", 
                                       audios=wf.lzin.x,
                                       n_mfcc=n_mfcc,
                                       n_fft=n_fft,
                                       n_mels=n_mels,
                                       win_length=win_length,
                                       hop_length=hop_length))
    wf.add(extract_mel_spectrogram_from_audios_pt(name="extract_mel_spectrogram_from_audios_pt", 
                                                  audios=wf.lzin.x,
                                                  n_mels=n_mels,
                                                  n_nfft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length
                                                  ))
    wf.add(extract_spectrogram_from_audios_pt(name="extract_spectrogram_from_audios_pt", 
                                              audios=wf.lzin.x,
                                              n_nfft=n_fft,
                                              win_length=win_length,
                                              hop_length=hop_length
                                              ))

    # setting multiple workflow outputs
    wf.set_output(
        [
            ("pitch_out", wf.extract_pitch_from_audios_pt.lzout.out),
            ("mel_filter_bank_out", wf.extract_mel_filter_bank_from_audios_pt.lzout.out),
            ("mfcc_out", wf.extract_mfcc_from_audios_pt.lzout.out),
            ("mel_spectrogram_out", wf.extract_mel_spectrogram_from_audios_pt.lzout.out),
            ("spectrogram_out", wf.extract_spectrogram_from_audios_pt.lzout.out),
        ]
    )

    with pydra.Submitter(plugin=plugin) as sub:
        sub(wf)

    outputs = wf.result()

    formatted_output: List[Dict[str, Any]] = []
    for output in outputs:
        formatted_output_item = {
            "torchaudio": {
                "pitch": output.output.pitch_out[0]["pitch"],
                "mel_filter_bank": output.output.mel_filter_bank_out[0]["mel_filter_bank"],
                "mfcc": output.output.mfcc_out[0]["mfcc"],
                "mel_spectrogram": output.output.mel_spectrogram_out[0]["mel_spectrogram"],
                "spectrogram": output.output.spectrogram_out[0]["spectrogram"],
            }
        }

        formatted_output.append(formatted_output_item)

    return formatted_output