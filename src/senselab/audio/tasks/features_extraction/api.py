"""High-level feature extraction for audio.

This module aggregates multiple feature backends—OpenSMILE, Praat/Parselmouth,
torchaudio, and torchaudio-squim—into a single convenience function. Each
backend can be toggled on/off or configured independently.

Backends:
    - **OpenSMILE**: Robust, hand-crafted descriptors (e.g., eGeMAPS).
    - **Praat/Parselmouth**: Prosody and voice-quality measures
        (pitch, jitter, shimmer, formants, etc.).
    - **torchaudio**: Spectral features (spectrogram, mel, MFCC, pitch).
    - **torchaudio-squim**: Objective quality metrics (e.g., STOI, PESQ, SI-SDR).
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union

from joblib import Memory, Parallel, delayed

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType
from senselab.utils.data_structures.logging import logger

from .opensmile import extract_opensmile_features_from_audios
from .ppg import extract_ppgs_from_audios
from .praat_parselmouth import extract_praat_parselmouth_features_from_audios
from .sparc import SparcFeatureExtractor
from .torchaudio import extract_torchaudio_features_from_audios
from .torchaudio_squim import extract_objective_quality_features_from_audios


def extract_features_from_audios(
    audios: List[Audio],
    opensmile: Union[Dict[str, str], bool] = True,
    parselmouth: Union[Dict[str, str], bool] = True,
    torchaudio: Union[Dict[str, str], bool] = True,
    torchaudio_squim: bool = True,
    sparc: Optional[bool] = None,
    ppgs: Optional[bool] = None,
    device: Optional[DeviceType] = None,
    n_jobs: int = 1,
    backend: Literal["threading", "loky", "multiprocessing", "sequential"] = "sequential",
    verbose: int = 0,
    cache_dir: Optional[str | os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Extract multi-backend features for each `Audio` and return a list of dicts.

    Enabled joblib backends run in parallelizable sub-workflows (where applicable)
    and their outputs are merged per audio. Disable any backend by passing ``False``;
    customize a backend by passing a dict (see below for keys and defaults).

    Args:
        audios (list[Audio]):
            Input audio objects.
        opensmile (dict | bool, optional):
            - ``False`` → skip OpenSMILE.
            - ``True``  → use defaults:
                ``{"feature_set": "eGeMAPSv02", "feature_level": "Functionals"}``
            - ``dict`` → override any of the above keys. `feature_set` and `feature_level`
              should match OpenSMILE presets.
        parselmouth (dict | bool, optional):
            - ``False`` → skip Praat/Parselmouth.
            - ``True``  → use defaults (pitch, intensity, jitter, shimmer, formants, etc. enabled):
                ``{"time_step": 0.005, "window_length": 0.025, "pitch_unit": "Hertz",
                  "speech_rate": True, "intensity_descriptors": True,
                  "harmonicity_descriptors": True, "formants": True, "spectral_moments": True,
                  "pitch": True, "slope_tilt": True, "cpp_descriptors": True, "duration": True,
                  "jitter": True, "shimmer": True, "n_jobs": 1, "backend": "loky", "verbose": 0,
                  "cache_dir": None}``
            - ``dict`` → override any of the above keys.
        torchaudio (dict | bool, optional):
            - ``False`` → skip torchaudio.
            - ``True``  → use defaults:
                ``{"freq_low": 80, "freq_high": 500, "n_fft": 1024, "n_mels": 128,
                  "n_mfcc": 40, "win_length": None, "hop_length": None}``
            - ``dict`` → override any of the above keys (e.g., ``n_fft``, ``hop_length``).
        torchaudio_squim (bool, optional):
            - ``False`` → skip objective quality metrics.
            - ``True``  → compute metrics such as STOI, PESQ, SI-SDR (backend-dependent defaults).
        sparc (bool, optional):
            - ``False`` → skip Speech Articulatory Coding features.
            - ``True`` → use the SPARC encoding model to generate features (currently generate and return all)
        ppgs (bool, optional):
            - ``False`` → skip Phonetic Posteriorgrams extrtaction.
            - ``True`` → use the ppgs model to generate phonetic posteriorgrams
        device (DeviceType, optional):
            Device to run feature extractions on for features where GPU might enhance performance (`torchaudio_squim`)
        n_jobs (int, optional):
            Number of parallel jobs to run (default: 1).
        backend (str, optional):
            Backend to use for parallelization.
            - “sequential” (used by default) is a serial backend.
            - “loky” can induce some communication and memory overhead
            when exchanging input and output data with the worker Python processes.
            On some rare systems (such as Pyiodide), the loky backend may not be available.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool.
            Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from
            the Python Global Interpreter Lock if the called function relies
            a lot on Python objects. “threading” is mostly useful when the execution
            bottleneck is a compiled extension that explicitly releases the GIL
            (for instance a Cython loop wrapped in a “with nogil” block or an expensive
            call to a library such as NumPy).
        verbose (int, optional):
            Verbosity (default: 0).
            If non zero, progress messages are printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        cache_dir (str | os.PathLike, optional):
            Path to cache directory. If None is given, no caching is done.

    Returns:
        list[dict[str, Any]]: One dict per input audio. Keys present depend on
        enabled backends; typical structure:

        - ``"opensmile"`` → ``dict[str, float]`` of aggregated descriptors.
        - ``"praat_parselmouth"`` → ``dict[str, float]`` (prosody/voice-quality).
        - ``"torchaudio"`` → nested ``dict[str, Tensor]`` (e.g., ``spectrogram``,
          ``mel_spectrogram``, ``mfcc``, ``pitch``). Tensors have shapes defined
          by your STFT/mel/MFCC settings.
        - ``"torchaudio_squim"`` → ``dict[str, float]`` with objective quality scores.
        - ``"sparc"`` → ``dict[str, Tensor]``
        - ``"ppgs"`` → Tensor

    Raises:
        ModuleNotFoundError:
            If a requested backend library is not installed (e.g., `opensmile`,
            `praat-parselmouth`, or dependencies required by torchaudio-squim).
        ValueError:
            If invalid parameter combinations are provided to a backend.

    Tips:
        - **Memory**: Torchaudio tensors (spectrograms, mels) can be large. Convert or
          downsample if you only need summary stats.

    Example (all defaults):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> feats = extract_features_from_audios([a1])
        >>> sorted(feats[0].keys())
        ['opensmile', 'praat_parselmouth', 'torchaudio', 'torchaudio_squim', 'sparc', 'ppgs']

    Example (all defaults II):
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.features_extraction import extract_features_from_audios
        >>> from pathlib import Path
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> extract_features_from_audios([a1])
        [{'opensmile': {'F0semitoneFrom27.5Hz_sma3nz_amean': 25.710796356201172,
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': 0.1605353206396103,
        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0': 21.095951080322266,
        'F0semitoneFrom27.5Hz_sma3nz_percentile50.0': 25.9762020111084,
        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0': 29.512413024902344,
        'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2': 8.416461944580078,
        'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope': 82.34796905517578,
        'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope': 99.20043182373047,
        'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope': 22.002275466918945,
        'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope': 9.043970108032227,
        'loudness_sma3_amean': 0.86087566614151,
        'loudness_sma3_stddevNorm': 0.43875235319137573,
        'loudness_sma3_percentile20.0': 0.5877408981323242,
        'loudness_sma3_percentile50.0': 0.8352401852607727,
        'loudness_sma3_percentile80.0': 1.1747918128967285,
        'loudness_sma3_pctlrange0-2': 0.5870509147644043,
        'loudness_sma3_meanRisingSlope': 10.285204887390137,
        'loudness_sma3_stddevRisingSlope': 7.544795513153076,
        'loudness_sma3_meanFallingSlope': 7.612527370452881,
        'loudness_sma3_stddevFallingSlope': 4.15903902053833,
        'spectralFlux_sma3_amean': 0.3213598430156708,
        'spectralFlux_sma3_stddevNorm': 0.6921582818031311,
        'mfcc1_sma3_amean': 10.274803161621094,
        'mfcc1_sma3_stddevNorm': 1.1581648588180542,
        'mfcc2_sma3_amean': 4.262020111083984,
        'mfcc2_sma3_stddevNorm': 2.052302837371826,
        'mfcc3_sma3_amean': 7.624598026275635,
        'mfcc3_sma3_stddevNorm': 1.4570358991622925,
        'mfcc4_sma3_amean': 3.6676177978515625,
        'mfcc4_sma3_stddevNorm': 2.6902272701263428,
        'jitterLocal_sma3nz_amean': 0.019597552716732025,
        'jitterLocal_sma3nz_stddevNorm': 0.9063860177993774,
        'shimmerLocaldB_sma3nz_amean': 1.264746069908142,
        'shimmerLocaldB_sma3nz_stddevNorm': 0.4629262685775757,
        'HNRdBACF_sma3nz_amean': 3.6400067806243896,
        'HNRdBACF_sma3nz_stddevNorm': 0.5911334753036499,
        'logRelF0-H1-H2_sma3nz_amean': 1.215877652168274,
        'logRelF0-H1-H2_sma3nz_stddevNorm': 3.883843183517456,
        'logRelF0-H1-A3_sma3nz_amean': 18.830764770507812,
        'logRelF0-H1-A3_sma3nz_stddevNorm': 0.30870768427848816,
        'F1frequency_sma3nz_amean': 665.1713256835938,
        'F1frequency_sma3nz_stddevNorm': 0.41958823800086975,
        'F1bandwidth_sma3nz_amean': 1300.2757568359375,
        'F1bandwidth_sma3nz_stddevNorm': 0.16334553062915802,
        'F1amplitudeLogRelF0_sma3nz_amean': -132.1533660888672,
        'F1amplitudeLogRelF0_sma3nz_stddevNorm': -0.6691396832466125,
        'F2frequency_sma3nz_amean': 1657.013916015625,
        'F2frequency_sma3nz_stddevNorm': 0.17019854485988617,
        'F2bandwidth_sma3nz_amean': 1105.7457275390625,
        'F2bandwidth_sma3nz_stddevNorm': 0.24520403146743774,
        'F2amplitudeLogRelF0_sma3nz_amean': -132.76707458496094,
        'F2amplitudeLogRelF0_sma3nz_stddevNorm': -0.6468541026115417,
        'F3frequency_sma3nz_amean': 2601.6630859375,
        'F3frequency_sma3nz_stddevNorm': 0.11457356810569763,
        'F3bandwidth_sma3nz_amean': 1091.15087890625,
        'F3bandwidth_sma3nz_stddevNorm': 0.3787318468093872,
        'F3amplitudeLogRelF0_sma3nz_amean': -134.52105712890625,
        'F3amplitudeLogRelF0_sma3nz_stddevNorm': -0.620308518409729,
        'alphaRatioV_sma3nz_amean': -8.626543045043945,
        'alphaRatioV_sma3nz_stddevNorm': -0.4953792095184326,
        'hammarbergIndexV_sma3nz_amean': 16.796842575073242,
        'hammarbergIndexV_sma3nz_stddevNorm': 0.3567312955856323,
        'slopeV0-500_sma3nz_amean': 0.021949246525764465,
        'slopeV0-500_sma3nz_stddevNorm': 1.0097224712371826,
        'slopeV500-1500_sma3nz_amean': -0.008139753714203835,
        'slopeV500-1500_sma3nz_stddevNorm': -1.6243411302566528,
        'spectralFluxV_sma3nz_amean': 0.4831695556640625,
        'spectralFluxV_sma3nz_stddevNorm': 0.48576226830482483,
        'mfcc1V_sma3nz_amean': 20.25444793701172,
        'mfcc1V_sma3nz_stddevNorm': 0.44413772225379944,
        'mfcc2V_sma3nz_amean': 3.619405746459961,
        'mfcc2V_sma3nz_stddevNorm': 2.1765975952148438,
        'mfcc3V_sma3nz_amean': 7.736487865447998,
        'mfcc3V_sma3nz_stddevNorm': 1.8630998134613037,
        'mfcc4V_sma3nz_amean': 4.60503625869751,
        'mfcc4V_sma3nz_stddevNorm': 2.864668846130371,
        'alphaRatioUV_sma3nz_amean': -2.5990121364593506,
        'hammarbergIndexUV_sma3nz_amean': 8.862899780273438,
        'slopeUV0-500_sma3nz_amean': 0.002166695659980178,
        'slopeUV500-1500_sma3nz_amean': 0.006735736038535833,
        'spectralFluxUV_sma3nz_amean': 0.24703539907932281,
        'loudnessPeaksPerSec': 3.8834950923919678,
        'VoicedSegmentsPerSec': 2.745098114013672,
        'MeanVoicedSegmentLengthSec': 0.12214285880327225,
        'StddevVoicedSegmentLengthSec': 0.09025190770626068,
        'MeanUnvoicedSegmentLength': 0.20666664838790894,
        'StddevUnvoicedSegmentLength': 0.17666037380695343,
        'equivalentSoundLevel_dBp': -24.297256469726562},
        'torchaudio': {'pitch': tensor([484.8485, 484.8485, 470.5882, 372.0930, 340.4255, 320.0000, 296.2963,
                140.3509, 135.5932, 126.9841, 124.0310, 124.0310, 113.4752, 110.3448,
                110.3448, 108.8435, 105.9603, 108.8435, 110.3448, 113.4752, 113.4752,
                124.0310, 113.4752, 113.4752, 108.8435, 105.9603, 105.9603, 105.9603,
                106.6667, 105.9603, 105.9603, 104.5752, 104.5752, 104.5752, 104.5752,
                101.2658, 101.2658, 100.6289, 100.6289, 100.0000, 100.0000,  98.1595,
                    98.1595,  98.1595,  95.8084,  95.8084,  95.8084,  95.2381,  95.2381,
                    94.6746,  91.9540,  91.9540,  91.9540,  91.9540,  91.9540,  91.4286,
                    91.4286,  91.4286,  91.4286,  91.4286,  91.4286,  91.4286,  90.9091,
                    90.9091,  90.9091,  91.4286,  91.4286,  91.4286,  91.4286,  91.4286,
                    91.4286,  91.4286,  91.4286,  91.4286,  91.4286,  91.4286,  91.4286,
                    91.4286,  91.9540,  91.9540,  93.0233,  93.5673,  93.5673,  94.1176,
                    94.6746,  94.6746,  94.6746,  95.8084,  96.3855,  96.9697, 100.0000,
                100.0000, 100.0000, 100.0000, 100.0000, 100.0000, 103.8961, 104.5752,
                104.5752, 106.6667, 106.6667, 106.6667, 111.1111, 116.7883, 116.7883,
                116.7883, 118.5185, 121.2121, 121.2121, 121.2121, 121.2121, 121.2121,
                121.2121, 121.2121, 121.2121, 121.2121, 121.2121, 121.2121, 122.1374,
                123.0769, 123.0769, 125.9843, 125.9843, 125.9843, 123.0769, 123.0769,
                122.1374, 122.1374, 121.2121, 121.2121, 121.2121, 120.3008, 117.6471,
                117.6471, 117.6471, 108.8435, 108.1081, 106.6667, 105.2632, 105.2632,
                105.2632, 105.2632, 105.2632, 105.2632, 105.2632, 105.2632, 105.2632,
                108.8435, 105.2632, 105.2632, 105.2632, 105.2632, 105.2632, 105.2632,
                105.2632, 119.4030, 119.4030, 120.3008, 120.3008, 121.2121, 121.2121,
                121.2121, 121.2121, 121.2121, 121.2121, 121.2121, 121.2121, 122.1374,
                122.1374, 122.1374, 122.1374, 123.0769, 123.0769, 123.0769, 123.0769,
                123.0769, 123.0769, 120.3008, 120.3008, 120.3008, 119.4030, 113.4752,
                106.6667, 103.2258, 103.2258,  96.9697,  96.9697,  96.9697,  96.9697,
                    96.9697,  96.9697,  96.9697,  96.9697,  96.9697,  96.9697,  96.9697,
                    96.9697,  96.9697,  96.9697,  96.9697,  96.9697,  96.9697,  96.9697,
                    96.9697,  96.9697,  96.9697,  96.9697,  96.9697,  97.5610,  97.5610,
                    97.5610,  97.5610,  97.5610,  98.1595, 100.0000, 100.6289, 100.6289,
                100.6289, 100.6289, 101.2658, 101.2658, 101.2658, 101.2658, 101.2658,
                101.2658, 101.2658, 101.2658, 101.2658, 101.2658, 101.2658,  97.5610,
                    90.9091,  89.8876,  88.8889,  88.8889,  88.3978,  87.4317,  86.0215,
                    86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,
                    86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,
                    86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,  86.0215,
                    86.0215,  86.0215,  86.0215,  86.0215,  86.4865,  86.4865,  86.4865,
                    86.4865,  86.4865,  87.4317,  87.9121,  87.9121,  87.9121,  89.8876,
                    90.9091,  90.9091,  90.9091,  90.9091,  90.9091,  91.4286,  91.4286,
                    91.4286,  92.4855,  92.4855,  93.0233,  93.0233,  93.0233,  93.5673,
                    93.5673,  95.2381,  95.2381, 100.0000, 101.9108, 112.6761, 112.6761,
                112.6761, 122.1374, 122.1374, 122.1374, 130.0813, 126.9841, 126.9841,
                130.0813, 130.0813, 130.0813, 130.0813, 137.9310, 130.0813, 130.0813,
                130.0813, 126.9841, 125.9843, 126.9841, 125.9843, 125.9843, 125.9843,
                125.9843, 125.9843, 126.9841, 126.9841, 130.0813, 130.0813, 126.9841,
                130.0813, 130.0813, 132.2314, 130.0813, 130.0813, 132.2314, 134.4538,
                134.4538, 135.5932, 135.5932, 137.9310, 135.5932, 135.5932, 135.5932,
                135.5932, 137.9310, 137.9310, 140.3509, 141.5929, 141.5929, 141.5929,
                144.1441, 144.1441, 149.5327, 149.5327, 149.5327, 141.5929, 141.5929,
                141.5929, 149.5327, 149.5327, 153.8462, 160.0000, 160.0000, 160.0000,
                160.0000, 160.0000, 163.2653, 164.9485, 164.9485, 164.9485, 164.9485,
                164.9485, 164.9485, 164.9485, 164.9485, 164.9485, 164.9485, 164.9485,
                164.9485, 164.9485, 164.9485, 164.9485, 156.8627, 155.3398, 155.3398,
                155.3398, 153.8462, 153.8462, 152.3810, 152.3810, 149.5327, 148.1481,
                148.1481, 148.1481, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890,
                148.1481, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890,
                146.7890, 146.7890, 145.4545, 145.4545, 152.3810, 153.8462, 153.8462,
                153.8462, 153.8462, 153.8462, 153.8462, 153.8462, 153.8462, 153.8462,
                153.8462, 153.8462, 153.8462, 153.8462, 153.8462, 153.8462, 153.8462,
                153.8462, 153.8462, 153.8462, 153.8462, 152.3810, 152.3810, 152.3810,
                152.3810, 149.5327, 148.1481, 148.1481, 148.1481, 148.1481, 148.1481,
                148.1481, 146.7890, 148.1481, 148.1481, 145.4545, 145.4545, 145.4545,
                145.4545, 145.4545, 144.1441, 144.1441, 144.1441, 142.8571, 142.8571,
                142.8571, 142.8571, 142.8571, 142.8571, 142.8571, 144.1441, 144.1441,
                145.4545, 145.4545, 145.4545, 145.4545, 146.7890, 146.7890, 146.7890,
                146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890,
                146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 145.4545,
                145.4545, 145.4545, 145.4545, 145.4545, 145.4545, 145.4545, 145.4545,
                146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890, 146.7890,
                400.0000, 400.0000, 484.8485, 484.8485, 484.8485, 484.8485, 484.8485,
                484.8485, 484.8485, 484.8485, 484.8485, 484.8485]),
        'mel_filter_bank': tensor([[4.4167e-04, 1.0165e-02, 1.3079e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [3.0977e-04, 1.5698e-02, 1.5785e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [8.2318e-05, 1.4367e-02, 2.8095e-01,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                ...,
                [3.6322e-05, 9.7330e-03, 5.4812e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [2.2802e-05, 1.2481e-02, 5.8374e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [5.3029e-05, 3.1305e-02, 7.9842e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00]]),
        'mfcc': tensor([[-6.2570e+02, -4.7505e+02, -3.1078e+02,  ..., -6.3893e+02,
                    -6.3893e+02, -6.3893e+02],
                [ 1.3593e+01,  1.9928e+01,  2.6022e+01,  ...,  3.9824e-05,
                    3.9824e-05,  3.9824e-05],
                [ 7.3933e+00, -2.1680e+01, -1.4259e+01,  ..., -1.3440e-05,
                    -1.3440e-05, -1.3440e-05],
                ...,
                [ 1.8122e+00, -3.1072e+00, -3.7336e+00,  ...,  7.0669e-05,
                    7.0669e-05,  7.0669e-05],
                [-2.7518e-01, -9.4738e+00, -2.3157e+00,  ..., -1.7963e-04,
                    -1.7963e-04, -1.7963e-04],
                [ 2.3144e-01, -6.4129e+00, -8.4420e+00,  ..., -1.5891e-04,
                    -1.5891e-04, -1.5891e-04]]),
        'mel_spectrogram': tensor([[4.4167e-04, 1.0165e-02, 1.3079e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [3.0977e-04, 1.5698e-02, 1.5785e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [8.2318e-05, 1.4367e-02, 2.8095e-01,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                ...,
                [3.6322e-05, 9.7330e-03, 5.4812e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [2.2802e-05, 1.2481e-02, 5.8374e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [5.3029e-05, 3.1305e-02, 7.9842e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00]]),
        'spectrogram': tensor([[3.5553e-06, 5.9962e-03, 2.7176e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [5.0707e-04, 1.1670e-02, 1.5016e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [3.1901e-04, 1.8529e-02, 1.8078e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                ...,
                [1.0302e-05, 3.5917e-03, 2.7169e-03,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [9.6637e-08, 1.3364e-03, 1.8495e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00],
                [1.4414e-05, 1.0598e-04, 2.8004e-02,  ..., 0.0000e+00, 0.0000e+00,
                    0.0000e+00]])},
        'parselmouth': ({'duration': 5.1613125,
            'speaking_rate': 3.874983349680919,
            'articulation_rate': 3.874983349680919,
            'phonation_ratio': 1.0,
            'pause_rate': 0.0,
            'mean_pause_duration': 0.0,
            'mean_f0_hertz': 118.59917806814313,
            'std_f0_hertz': 30.232960797931817,
            'mean_intensity_db': 69.76277128148347,
            'std_intensity_db': 58.54414165935646,
            'range_ratio_intensity_db': -0.25736445047981316,
            'pitch_floor': 60.0,
            'pitch_ceiling': 250.0,
            'mean_hnr_db': 3.3285614070654375,
            'std_hnr_db': 3.36490968797237,
            'spectral_slope': -13.982306776816046,
            'spectral_tilt': -0.004414961849917737,
            'cepstral_peak_prominence_mean': 7.0388038514346825,
            'cepstral_peak_prominence_std': 1.5672438573255245,
            'mean_f1_loc': 613.4664268420964,
            'std_f1_loc': 303.98235579059883,
            'mean_b1_loc': 401.96960219300837,
            'std_b1_loc': 400.9001719378358,
            'mean_f2_loc': 1701.7755281579418,
            'std_f2_loc': 325.4405394017738,
            'mean_b2_loc': 434.542188503193,
            'std_b2_loc': 380.8914612651878,
            'spectral_gravity': 579.587511962247,
            'spectral_std_dev': 651.3025011919739,
            'spectral_skewness': 3.5879707548251045,
            'spectral_kurtosis': 19.991495997865282,
            'local_jitter': 0.02553484151620524,
            'localabsolute_jitter': 0.00021392842618599855,
            'rap_jitter': 0.012174051087556429,
            'ppq5_jitter': 0.01597797849248675,
            'ddp_jitter': 0.03652215326266929,
            'local_shimmer': 0.1530474665829716,
            'localDB_shimmer': 1.3511061323188314,
            'apq3_shimmer': 0.0702984931637734,
            'apq5_shimmer': 0.09680154282272849,
            'apq11_shimmer': 0.19065409516266155,
            'dda_shimmer': 0.2108954794913202},),
        'torchaudio_squim': {
            'stoi': 0.9247563481330872,
            'pesq': 1.3702949285507202,
            'si_sdr': 11.71167278289795
            },
        'ppgs': tensor([[[3.0436e-08, 2.7577e-08, 2.3042e-08,  ..., 2.7873e-05,
            3.1141e-04, 3.2798e-03],
            [1.8068e-08, 1.8751e-08, 1.8935e-08,  ..., 4.1033e-05,
            4.0143e-04, 3.7301e-03],
            [3.2914e-07, 2.9896e-07, 2.7825e-07,  ..., 2.8734e-04,
            2.0516e-03, 1.4649e-02],
            ...,
            [1.1213e-05, 1.1479e-05, 1.0012e-05,  ..., 1.5745e-04,
            1.2167e-03, 8.9809e-03],
            [1.9514e-09, 2.1018e-09, 1.9591e-09,  ..., 2.6614e-07,
            7.8304e-06, 1.9822e-04],
            [9.9998e-01, 9.9998e-01, 9.9998e-01,  ..., 9.9360e-01,
            9.5824e-01, 7.3107e-01]]]),
        'sparc': {
            'ema': array([[-0.0569168 ,  0.05923977, -0.6384876 , ..., -1.2920734 ,
                    -1.112251  ,  0.93640935],
                [ 0.46935433, -0.6915469 , -0.04576159, ..., -1.4056824 ,
                    -0.56052697,  0.87104446],
                [ 0.80757684, -1.0827653 ,  0.36380562, ..., -1.2541789 ,
                    -0.13409577,  0.6728183 ],
                ...,
                [-0.13434991,  0.20647803, -1.0112027 , ..., -1.0819684 ,
                    0.22669971,  0.48441225],
                [ 0.02813378,  0.29047868, -0.77194595, ..., -1.1423542 ,
                    0.2727234 ,  0.3466994 ],
                [ 0.30194056,  0.11315   , -0.04662162, ..., -0.6694075 ,
                    0.07024295,  0.11801762]], shape=(2106, 12), dtype=float32),
            'loudness': array([[0.0211054 ],
                [0.03673043],
                [0.01156581],
                ...,
                [0.23418687],
                [0.15966089],
                [0.0361287 ]], shape=(2107, 1), dtype=float32),
        'pitch': array([[115.200165],
            [114.987274],
            [116.067245],
            ...,
            [161.98538 ],
            [160.74951 ],
            [161.75674 ]], shape=(2107, 1), dtype=float32),
        'periodicity': array([[0.00023171],
            [0.02182695],
            [0.02822398],
            ...,
            [0.00497394],
            [0.00319474],
            [0.00147732]], shape=(2107, 1), dtype=float32),
        'pitch_stats': array([127.39656 ,  12.342598], dtype=float32),
        'spk_emb': array([-0.3200726 ,  0.6454727 , -0.33973247, -0.05436979, -0.75786656,
                0.21105185,  1.0232798 , -0.8574615 ,  0.25824642,  0.01725103,
            -1.1606296 ,  0.2200551 , -0.5214666 , -0.53481615, -0.4383874 ,
            -0.38553894, -0.41577503, -0.5511529 ,  0.4387015 , -0.13225996,
                0.7347815 , -0.2443064 , -0.00733899,  0.03008379, -0.3554219 ,
            -1.0962536 , -0.55057245,  0.5695376 ,  0.7131355 ,  0.4725528 ,
            -0.22333047,  0.0072331 ,  0.6192075 , -0.5492355 , -1.6081889 ,
            -0.05475338,  0.31899607,  0.23982322,  0.2328668 ,  0.46595818,
                0.28324106, -0.74834126, -0.18845464,  0.05998179, -0.47815633,
                0.16281566,  0.5345152 ,  1.6304512 , -0.6535251 , -0.18741362,
                0.7163336 , -0.5160271 ,  0.17237644,  0.14135993,  0.0999315 ,
            -1.0836225 ,  1.396034  ,  0.05934188,  1.1151409 ,  0.34214428,
            -0.14974377, -0.6432309 , -0.14783166,  0.21114561], dtype=float32), 'ft_len': np.int64(2107)}
        }]

    Example (disable OpenSMILE; customize torchaudio):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> feats = extract_features_from_audios(
        ...     [a1],
        ...     opensmile=False,
        ...     torchaudio={
        ...         "n_fft": 2048,
        ...         "hop_length": 256
        ...     },
        ... )
        >>> "opensmile" in feats[0]
        False

    Example (Parselmouth only, custom pitch range):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> feats = extract_features_from_audios(
        ...     [a1],
        ...     opensmile=False,
        ...     torchaudio=False,
        ...     torchaudio_squim=False,
        ...     parselmouth={"pitch_unit": "Hertz"},
        ...     ppgs=False,
        ...     sparc=False
        ... )
        >>> "praat_parselmouth" in feats[0]
        True
    """
    # defaults
    default_opensmile: Dict[str, Any] = {"feature_set": "eGeMAPSv02", "feature_level": "Functionals"}
    default_parselmouth: Dict[str, Any] = {
        "time_step": 0.005,
        "window_length": 0.025,
        "pitch_unit": "Hertz",
        "speech_rate": True,
        "intensity_descriptors": True,
        "harmonicity_descriptors": True,
        "formants": True,
        "spectral_moments": True,
        "pitch": True,
        "slope_tilt": True,
        "cpp_descriptors": True,
        "duration": True,
        "jitter": True,
        "shimmer": True,
        "n_jobs": 1,
        "backend": "loky",
        "verbose": 0,
        "cache_dir": None,
    }
    default_torchaudio: Dict[str, Any] = {
        "freq_low": 80,
        "freq_high": 500,
        "n_fft": 1024,
        "n_mels": 128,
        "n_mfcc": 40,
        "win_length": None,
        "hop_length": None,
    }

    # Resolve configs
    use_opensmile = bool(opensmile)
    use_parselmouth = bool(parselmouth)
    use_torchaudio = bool(torchaudio)
    use_squim = bool(torchaudio_squim)

    if sparc is None or ppgs is None:
        logger.warning(
            "Future major versions of senselab will default `sparc` and `ppgs` to True. \
            None values will not be allowed. To maintain current functionality, please set these to False."
        )

    use_sparc = bool(sparc)
    use_ppgs = bool(ppgs)

    if not any([use_opensmile, use_parselmouth, use_torchaudio, use_squim, use_sparc, use_ppgs]):
        return [{} for _ in audios]

    def _extract_one(a: Audio) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if use_opensmile:
            my_opensmile = {**default_opensmile, **(opensmile if isinstance(opensmile, dict) else {})}
            out["opensmile"] = extract_opensmile_features_from_audios([a], **my_opensmile)[0]
        if use_parselmouth:
            my_parselmouth = {**default_parselmouth, **(parselmouth if isinstance(parselmouth, dict) else {})}
            out["praat_parselmouth"] = extract_praat_parselmouth_features_from_audios([a], **my_parselmouth)[0]
        if use_torchaudio:
            my_ta = {**default_torchaudio, **(torchaudio if isinstance(torchaudio, dict) else {})}
            out["torchaudio"] = extract_torchaudio_features_from_audios([a], **my_ta)[0]
        if use_squim:
            out["torchaudio_squim"] = extract_objective_quality_features_from_audios([a])[0]
        if use_sparc:
            out["sparc"] = SparcFeatureExtractor.extract_sparc_features([a], device=device, resample=True)[0]
        if use_ppgs:
            out["ppgs"] = extract_ppgs_from_audios([a], device=device)[0]
        return out

    # Cache
    memory: Optional[Memory] = Memory(str(cache_dir), verbose=verbose) if cache_dir else None
    if memory:
        _extract_one = memory.cache(_extract_one)

    # Parallel across audios
    return Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
    )(delayed(_extract_one)(a) for a in audios)
