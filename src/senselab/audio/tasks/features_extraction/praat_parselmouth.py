"""This module contains functions that extract features from audio files using the PRAAT library.

The initial implementation of this features extraction was started by Nicholas Cummins
from King's College London and has since been further developed and maintained
by the senselab community.
"""

import inspect
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from joblib import Memory, Parallel, delayed

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import logger

try:
    import parselmouth  # type: ignore

    PARSELMOUTH_AVAILABLE = True
except ModuleNotFoundError:
    PARSELMOUTH_AVAILABLE = False

    class DummyParselmouth:
        """Dummy class for when parselmouth is not available.

        This is helpful for type checking when parselmouth is not installed.
        """

        def __init__(self) -> None:
            """Dummy constructor for when parselmouth is not available."""
            pass

        def call(self, *args: object, **kwargs: object) -> None:  # type: ignore
            """Dummy method for when parselmouth is not available."""

        class Sound:
            """Dummy class for when parselmouth is not available."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Dummy class for when parselmouth is not available."""
                pass

    parselmouth = DummyParselmouth()


PITCH_FLOOR_PERCENTILE = 5.0  # feeds only the floor's ratio term
PITCH_CEILING_QUARTILE = 75.0  # feeds only the ceiling's first ratio term

# Library defaults for the five narrowing coefficients. The triage path passes
# `praat_features.pitch_*` instead; see specs/20260817-triage-workflow-dag/config-derivations.md.
DEFAULT_PITCH_FLOOR_DIVISOR = 1.5  # a ratio, not an octave span
DEFAULT_PITCH_CEILING_QUARTILE_MULTIPLIER = 2.5  # a ratio, not an octave span
DEFAULT_PITCH_PINNED_PERCENTILE = 95.0  # the branch predicate's statistic and the excursion term's
DEFAULT_PITCH_EXCURSION_MULTIPLIER = 1.5  # a ratio, not an octave span
DEFAULT_PITCH_PINNED_OCTAVE_RATIO = 2.0  # one octave above the search floor

# Structural conventions of the cepstrogram, not coefficients that scale it. They name how the
# analysis window and the robust fit are *shaped*; the values that move a CPPS are `CppsSettings`.
CPPS_WINDOW_PERIODS = 3.0  # effective analysis width, in periods of the cepstrogram pitch floor
CPPS_GAUSSIAN_WIDTH_FACTOR = 2.0  # physical Gaussian duration, as a multiple of that width
CPPS_HUBER_K = 1.345  # Huber's tuning constant, 95% efficiency at the Gaussian
CPPS_MAX_ROBUST_ITERATIONS = 50  # iteration cap for the reweighting, independent of the tolerance
CPPS_QUEFRENCY_FLOOR = 1e-300  # guards log10 of an exactly-zero cepstral bin


@dataclass(frozen=True)
class CppsSettings:
    """Every setting a smoothed cepstral peak prominence is computed under.

    In triage each field is read from the ``praat_features.cpps`` config section; the defaults here
    are the library's. Their derivations are in
    ``specs/20260817-triage-workflow-dag/config-derivations.md``.

    Attributes:
        pitch_floor_hz: Sets the cepstrogram's analysis window, whose effective width is
            ``CPPS_WINDOW_PERIODS / pitch_floor_hz``.
        time_step_s: Hop between cepstrogram frames.
        max_frequency_hz: Upper edge of the analysed band; the signal is resampled to twice it.
        preemphasis_from_hz: Pre-emphasis corner applied before framing.
        time_averaging_s: First smoothing window, across frames.
        quefrency_averaging_s: Second smoothing window, across quefrency within a frame.
        peak_search_floor_hz: Lowest F0 the peak is searched for.
        peak_search_ceiling_hz: Highest F0 the peak is searched for.
        trend_start_s: Lowest quefrency the trend line is fitted over.
        trend_end_s: Highest quefrency it is fitted over; ``0.0`` means the end of the axis.
        robust_tolerance: Relative slope change below which the reweighting has converged.
        subtract_tilt_before_smoothing: Whether the trend is removed before smoothing. Only
            ``False`` is implemented.
        tilt_line_type: Shape of the trend line. Only ``"straight"`` is implemented.
        peak_interpolation: How the peak is refined between bins. Only ``"parabolic"`` is
            implemented.
    """

    pitch_floor_hz: float = 60.0
    time_step_s: float = 0.002
    max_frequency_hz: float = 5000.0
    preemphasis_from_hz: float = 50.0
    time_averaging_s: float = 0.01
    quefrency_averaging_s: float = 0.001
    peak_search_floor_hz: float = 60.0
    peak_search_ceiling_hz: float = 700.0
    trend_start_s: float = 0.001
    trend_end_s: float = 0.0
    robust_tolerance: float = 0.05
    subtract_tilt_before_smoothing: bool = False
    tilt_line_type: str = "straight"
    peak_interpolation: str = "parabolic"


DEFAULT_CPPS_SETTINGS = CppsSettings()
"""The library defaults; the triage path passes ``praat_features.cpps`` instead."""


def get_sound(audio: Union[Path, Audio], sampling_rate: int = 16000) -> parselmouth.Sound:
    """Get a sound object from a given audio file or Audio object.

    Args:
        audio (Union[Path, Audio]): A path to an audio file or an Audio object.
        sampling_rate (int, optional): The sampling rate of the audio. Defaults to 16000.

    Returns:
        parselmouth.Sound: A Parselmouth Sound object.

    Raises:
        FileNotFoundError: If the file is not found at the given path.
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        # Loading the sound
        if isinstance(audio, Path):
            audio = audio.resolve()
            if not audio.exists():
                logger.error(f"File does not exist: {audio}")
                raise FileNotFoundError(f"File does not exist: {audio}")
            snd_full = parselmouth.Sound(str(audio))
        elif isinstance(audio, Audio):
            snd_full = parselmouth.Sound(audio.waveform, audio.sampling_rate)

        # Preprocessing
        if snd_full.n_channels > 1:
            snd_full = snd_full.convert_to_mono()
        if snd_full.sampling_frequency != sampling_rate:
            snd_full = parselmouth.praat.call(snd_full, "Resample", sampling_rate, 50)
            # Details of queery: https://www.fon.hum.uva.nl/praat/manual/Get_sampling_frequency.html
            # Details of conversion: https://www.fon.hum.uva.nl/praat/manual/Sound__Resample___.html
    except Exception as e:
        raise RuntimeError(f"Error loading sound: {e}")
    return snd_full


def extract_speech_rate(snd: Union[parselmouth.Sound, Path, Audio]) -> Dict[str, float]:
    """Extract speech timing and pausing features from a given sound object.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.

    Returns:
        Dict[str, float]: A dictionary containing the following features:

            - speaking_rate (float): Number of syllables divided by duration.
            - articulation_rate (float): Number of syllables divided by phonation time.
            - phonation_ratio (float): Phonation time divided by duration.
            - pause_rate (float): Number of pauses divided by duration.
            - mean_pause_dur (float): Total time pausing divided by the number of identified pauses.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_speech_rate(snd)
        {
            'speaking_rate': 5.3,
            'articulation_rate': 4.7,
            'phonation_ratio': 0.9,
            'pause_rate': 2.1,
            'mean_pause_dur': 0.5
        }
        ```

    Useful sources for this code:

        - https://sites.google.com/view/uhm-o-meter/scripts/syllablenuclei_v3?pli=1
        - https://drive.google.com/file/d/1o3mNdN5FKTiYQC9GHB1XoZ8JJIGZk_AK/view
        - (2009 paper) https://doi.org/10.3758/BRM.41.2.385
        - (2021 paper) https://doi.org/10.1080/0969594X.2021.1951162
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        # _____________________________________________________________________________________________________________
        # Load the sound object into parselmouth if it is an Audio object
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # _____________________________________________________________________________________________________________
        # Key pause detection hyperparameters

        # Silence Threshold (dB) - standard setting to detect silence in the "To TextGrid (silences)" function.
        # The higher this number, the lower the chances of finding silent pauses
        silence_db = -25

        # Minimum_dip_between_peaks_(dB) - if there are decreases in intensity
        # of at least this value surrounding the peak, the peak is labelled to be a syllable nucleus
        # I.e. the size of the dip between two possible peakes
        # The higher this number, the less syllables will be found
        # For clean and filtered signal use 4, if not use 2 (recommend thresholds)
        min_dip = 4
        # Code for determining if the signal not clean/filtered
        hnr = parselmouth.praat.call(
            snd.to_harmonicity_cc(), "Get mean", 0, 0
        )  # Note: (0,0) is the time range for extraction, setting both two zero tells praat to use the full file
        if hnr < 60:
            min_dip = 2

        # Minimum pause duration (s): How long should a pause be to be counted as a silent pause?
        # The higher this number, the fewer pauses will be found
        min_pause = 0.3  # the default for this is 0.1 in Praat, the de Jong's script has this set at 0.3
        # Based on values in: Toward an understanding of fluency:
        # A microanalysis of nonnative speaker conversations (Riggenbach)
        # – Micropause (silence of .2s or less)
        # – Hesitation (silence of .3 to .4s)
        # – Unfilled pause (silence of .5s or more)

        # ______________________________________________________________________________________________________________
        # Intensity information

        intensity = snd.to_intensity(minimum_pitch=50, time_step=0.016, subtract_mean=True)
        # These are the setting recommended by de jong - minimum pitch” set to 50 Hz,.
        # With this parameter setting, we extract intensity smoothed over a time window of (3.2/minimum_pitch)=64 msec,
        #  with 16-msec time steps explanation on these calculations are found at:
        # https://www.fon.hum.uva.nl/praat/manual/Sound__To_Intensity___.html

        min_intensity = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "Parabolic")  # time range, Interpolation
        max_intensity = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "Parabolic")  # time range, Interpolation

        # Silince is detected by measuring whether the intensity is 25 dB below the 99% highest peak
        # 99% is chosen to eliminate short loud bursts in intensity that may not have been speech

        # get .99 quantile to get maximum (without influence of non-speech sound bursts)
        max_99_intensity = parselmouth.praat.call(intensity, "Get quantile", 0, 0, 0.99)

        # estimate Intensity threshold
        silence_db_1 = max_99_intensity + silence_db
        db_adjustment = max_intensity - max_99_intensity
        silence_db_2 = silence_db - db_adjustment
        if silence_db_1 < min_intensity:
            silence_db_1 = min_intensity

        # ______________________________________________________________________________________________________________
        # Create a TextGrid in which the silent and sounding intervals, store these intervals

        textgrid = parselmouth.praat.call(
            intensity, "To TextGrid (silences)", silence_db_2, min_pause, 0.1, "silent", "sounding"
        )
        # Hyperparameters:
        # Silence threshold (dB),
        # Minimum silent interval (s) - minimum duration for an interval to be considered as silent
        # Minimum sounding interval (s) - minimum duration for an interval to be not considered as silent
        # Silent interval label
        # Sounding interval label

        # Loop through intervals and extract times of identified silent and sounding sections
        silencetier = parselmouth.praat.call(textgrid, "Extract tier", 1)
        silencetable = parselmouth.praat.call(silencetier, "Down to TableOfReal", "sounding")
        npauses = parselmouth.praat.call(silencetable, "Get number of rows")

        phonation_time = 0
        for ipause in range(npauses):
            pause = ipause + 1
            beginsound = parselmouth.praat.call(silencetable, "Get value", pause, 1)
            endsound = parselmouth.praat.call(silencetable, "Get value", pause, 2)
            speakingdur = endsound - beginsound

            phonation_time += speakingdur

            # This is to remove the first (before first word) and last (after last word) silence from consideration
            if pause == 1:
                begin_speak = beginsound
            if pause == (npauses):
                end_speak = endsound

        # ______________________________________________________________________________________________________________
        # Next block of code finds all possible peaks

        # Convert intensity countor into sound representation
        intensity_matrix = parselmouth.praat.call(intensity, "Down to Matrix")  # convert intensity to 2d representation

        # Convert intensity countor into sound representation
        sound_from_intensity_matrix = parselmouth.praat.call(intensity_matrix, "To Sound (slice)", 1)

        # find positive extrema, maxima in sound_from_intensity_matrix, which correspond to steepest rises in Intensity;
        point_process = parselmouth.praat.call(
            sound_from_intensity_matrix,
            "To PointProcess (extrema)",
            "Left",
            "yes",
            "no",
            "Sinc70",
        )

        # estimate peak positions (all peaks)
        t = []
        numpeaks = parselmouth.praat.call(point_process, "Get number of points")
        for i in range(numpeaks):
            t.append(parselmouth.praat.call(point_process, "Get time from index", i + 1))

        # ______________________________________________________________________________________________________________
        # Find the time and values of all peaks

        # fill array with intensity values
        timepeaks = []
        peakcount = 0
        intensities = []
        for i in range(numpeaks):
            value = parselmouth.praat.call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
            if value > silence_db_1:
                peakcount += 1
                intensities.append(value)
                timepeaks.append(t[i])

        # ______________________________________________________________________________________________________________
        # Now find all valid peaks

        # fill array with valid peaks: only intensity values if preceding
        # dip in intensity is greater than min_dip
        validpeakcount = 0
        currenttime = timepeaks[0]
        currentint = intensities[0]
        validtime = []

        for p in range(peakcount - 1):
            following = p + 1
            followingtime = timepeaks[following]
            dip = parselmouth.praat.call(
                intensity, "Get minimum", currenttime, followingtime, "None"
            )  # Gets minimiun value between two time points, doesn't intepolote/filter
            diffint = abs(currentint - dip)
            if diffint > min_dip:
                validpeakcount += 1
                validtime.append(timepeaks[p])
            # Update current time and intensity values for next loop
            currenttime = timepeaks[following]
            currentint = parselmouth.praat.call(intensity, "Get value at time", timepeaks[following], "Cubic")

        # ______________________________________________________________________________________________________________
        # Extract voicing information

        pitch = snd.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
        # Praat page for hyperparamters https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html
        # From de Jong's 2009 paper - We extract the pitch contour, this time using a window size of 100 msec
        # and 20-msec time steps, and exclude all peaks that are unvoiced
        # Key Hyperparamter are different to praat recommended - can't find a reason for this
        # time_step: Optional[Positive[float]] = None,  - set per De jong's recommendation
        # pitch_floor: Positive[float] = 75.0 set per dejong recommendation - 3/30 gives 100ms
        # max_number_of_candidates: Positive[int] = 15 (can't find a reason for this value being lower)
        # very_accurate: bool = False,
        # silence_threshold: float = 0.03,
        # voicing_threshold: float = 0.45, (can't find a reason for this value being different)
        # octave_cost: float = 0.01,
        # octave_jump_cost: float = 0.35,
        # voiced_unvoiced_cost: float = 0.14, (can't find a reason for this value being different)
        # pitch_ceiling: Positive[float] = 600.0 (can't find a reason for this value being lower, might change to value
        # from pitch_value function)

        # ______________________________________________________________________________________________________________
        # Loop through valid peaks, count ones that are voiced (i.e., have valid pitch value at the same time)

        number_syllables = int(0)
        for time in range(validpeakcount):
            querytime = validtime[time]
            whichinterval = parselmouth.praat.call(textgrid, "Get interval at time", 1, querytime)
            whichlabel = parselmouth.praat.call(textgrid, "Get label of interval", 1, whichinterval)
            value = pitch.get_value_at_time(querytime)
            if not np.isnan(value):
                if whichlabel == "sounding":
                    number_syllables += 1

        # ______________________________________________________________________________________________________________
        # return results

        original_dur = end_speak - begin_speak

        speaking_rate = number_syllables / original_dur
        articulation_rate = number_syllables / phonation_time
        phonation_ratio = phonation_time / original_dur

        number_pauses = npauses - 1
        pause_time = original_dur - phonation_time

        pause_rate = number_pauses / original_dur
        mean_pause_dur = pause_time / number_pauses if number_pauses > 0 else 0.0

        return {
            "speaking_rate": speaking_rate,
            "articulation_rate": articulation_rate,
            "phonation_ratio": phonation_ratio,
            "pause_rate": pause_rate,
            "mean_pause_dur": mean_pause_dur,
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "speaking_rate": np.nan,
            "articulation_rate": np.nan,
            "phonation_ratio": np.nan,
            "pause_rate": np.nan,
            "mean_pause_dur": np.nan,
        }


def _no_pitch_range(*, failed: float = 0.0) -> Dict[str, float]:
    """The five-key shape every ``extract_pitch_values`` path returns when no range was derived."""
    return {
        "pitch_floor": np.nan,
        "pitch_ceiling": np.nan,
        "pitch_frames": 0.0,
        "pitch_failed": failed,
        "pitch_range_fell_back": 0.0,
    }


def extract_pitch_values(
    snd: Union[parselmouth.Sound, Path, Audio],
    search_floor_hz: float = 50.0,
    search_ceiling_hz: float = 600.0,
    *,
    pitch_floor_divisor: float = DEFAULT_PITCH_FLOOR_DIVISOR,
    pitch_ceiling_quartile_multiplier: float = DEFAULT_PITCH_CEILING_QUARTILE_MULTIPLIER,
    pitch_pinned_percentile: float = DEFAULT_PITCH_PINNED_PERCENTILE,
    pitch_excursion_multiplier: float = DEFAULT_PITCH_EXCURSION_MULTIPLIER,
    pitch_pinned_octave_ratio: float = DEFAULT_PITCH_PINNED_OCTAVE_RATIO,
) -> Dict[str, float]:
    """Derive this recording's own pitch range by narrowing a wide autocorrelation search.

    Runs one wide pass over ``[search_floor_hz, search_ceiling_hz]``, then narrows the floor to
    ``max(search_floor_hz, p5 / pitch_floor_divisor)`` and the ceiling to ``min(search_ceiling_hz,
    max(q3 * pitch_ceiling_quartile_multiplier, pinned * pitch_excursion_multiplier))``, off the
    linear-Hz percentiles of the voiced contour, where ``pinned`` is the
    ``pitch_pinned_percentile``-th. When ``pinned`` falls below ``pitch_pinned_octave_ratio`` times
    the search floor the unnarrowed search range is returned instead.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        search_floor_hz (float): Lowest pitch of the wide search the narrow range is derived from.
        search_ceiling_hz (float): Highest pitch of that wide search.
        pitch_floor_divisor (float): Divides the 5th percentile to give the floor. In triage, read it
            from ``praat_features.pitch_floor_divisor``.
        pitch_ceiling_quartile_multiplier (float): Multiplies the upper quartile in the ceiling's
            first term. In triage, read it from ``praat_features.pitch_ceiling_quartile_multiplier``.
        pitch_pinned_percentile (float): The percentile the branch predicate tests and the ceiling's
            second term multiplies — one statistic serving both. In triage, read it from
            ``praat_features.pitch_pinned_percentile``.
        pitch_excursion_multiplier (float): Multiplies that percentile in the ceiling's second term.
            In triage, read it from ``praat_features.pitch_excursion_multiplier``.
        pitch_pinned_octave_ratio (float): Multiple of the search floor that percentile must clear
            for the narrowing to be used at all. In triage, read it from
            ``praat_features.pitch_pinned_octave_ratio``.

    Returns:
        dict: Five float keys, on all three return paths:

            - pitch_floor (float): The lowest pitch value to use in future pitch extraction algorithms.
            - pitch_ceiling (float): The highest pitch value to use in future pitch extraction algorithms.
            - pitch_frames (float): Voiced frames the range rests on; 0.0 when none were placed.
            - pitch_failed (float): 1.0 when the analysis itself raised, 0.0 otherwise.
            - pitch_range_fell_back (float): 1.0 when the unnarrowed search range was returned.

        ``pitch_floor`` and ``pitch_ceiling`` are NaN when no range could be derived — either because
        the wide search placed no pitch, or because the analysis failed. ``pitch_failed`` separates
        those two.

    Notes:
        The two-pass structure and the quartile ceiling term follow Hirst 2011, "The analysis by
        synthesis of speech melody". The percentile floor, the excursion term and the pinned-contour
        fallback are senselab's own. Every coefficient's derivation is in
        ``specs/20260817-triage-workflow-dag/config-derivations.md`` under ``praat_features``, and the
        rule's own record is in ``praat-instrument-audit.md`` under step 2.

        Important: These values are used within other functions, they are not outputs of the full code.

        Different pitch extraction methods in Praat:

        - Cross-correlation (Praat default) vs auto-correlation pitch extraction:
        both are used in different functions below.
        - Cross-correlation is better than auto-correlation at finding period-level variation,
        such as jitter and shimmer, whereas auto-correlation is better at finding intended intonation contours.
        - [Discussion on this on a Praat Forum](https://groups.io/g/Praat-Users-List/topic/pitch_detection_ac_vs_cc/78829266?p=,,,20,0,0,0::recentpostdate/sticky,,,20,2,20,78829266,previd=1612369050729515119,nextid=1605568402827788039&previd=1612369050729515119&nextid=1605568402827788039)

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_pitch_values(snd)
        {'pitch_floor': 80.0, 'pitch_ceiling': 300.0, 'pitch_frames': 188.0, 'pitch_failed': 0.0,
         'pitch_range_fell_back': 0.0}
        ```
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        pitch_wide = snd.to_pitch_ac(time_step=0.005, pitch_floor=search_floor_hz, pitch_ceiling=search_ceiling_hz)
        # Other than values above, I'm using default hyperparamters
        # Details: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html

        # the voiced frames of the wide pass; unvoiced frames come back as 0
        pitch_values = pitch_wide.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values != 0]
        if pitch_values.size == 0:
            return _no_pitch_range()

        low, upper_quartile, high = np.percentile(
            pitch_values, [PITCH_FLOOR_PERCENTILE, PITCH_CEILING_QUARTILE, pitch_pinned_percentile]
        )
        if float(high) < pitch_pinned_octave_ratio * float(search_floor_hz):
            floor, ceiling, fell_back = float(search_floor_hz), float(search_ceiling_hz), 1.0
        else:
            floor = max(float(search_floor_hz), float(low) / pitch_floor_divisor)
            ceiling = min(
                float(search_ceiling_hz),
                max(
                    float(upper_quartile) * pitch_ceiling_quartile_multiplier,
                    float(high) * pitch_excursion_multiplier,
                ),
            )
            fell_back = 0.0

        return {
            "pitch_floor": floor,
            "pitch_ceiling": ceiling,
            "pitch_frames": float(pitch_values.size),
            "pitch_failed": 0.0,
            "pitch_range_fell_back": fell_back,
        }
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return _no_pitch_range(failed=1.0)


def extract_pitch_descriptors(
    snd: Union[parselmouth.Sound, Path, Audio],
    floor: float,
    ceiling: float,
    frame_shift: float = 0.005,
    unit: str = "Hertz",
) -> Dict[str, float]:
    """Extract Pitch Features.

    Function to extract key pitch features from a given sound object.
    This function uses the pitch_ac method as autocorrelation is better at finding intended intonation contours.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.
            Defaults to 0.005.
        unit (str, optional): The unit in which the pitch is returned. Defaults to "Hertz".
            Could be "semitones".

    Returns:
        dict: A dictionary containing the following keys:

            - mean_f0_{unit} (float): Mean pitch in {unit}.
            - stdev_f0_{unit} (float): Standard deviation in {unit}.

    Notes:
        - Uses pitch_ac as autocorrelation is better at finding intended intonation contours.
        - stdev_f0_semitone is used in DOI: 10.1080/02699200400008353, which used this as a marker for dysphonia.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_pitch_descriptors(snd, 75, 500, 0.01, "Hertz")
        {'mean_f0_hertz': 220.5, 'stdev_f0_hertz': 2.5}
        ```
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract pitch object
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        # Other than values above, I'm using default hyperparameters
        # Details: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html

        # Extract mean, median, and standard deviation
        mean_f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, unit)  # time range, units
        stdev_f0 = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, unit)

        # Return results
        return {f"mean_f0_{unit.lower()}": mean_f0, f"stdev_f0_{unit.lower()}": stdev_f0}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {f"mean_f0_{unit.lower()}": np.nan, f"stdev_f0_{unit.lower()}": np.nan}


def extract_intensity_descriptors(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, frame_shift: float
) -> Dict[str, float]:
    """Extract Intensity Features.

    Function to extract key intensity information from a given sound object.
    This function is based on default Praat code adapted to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new intensity value, typically set to 5 ms.

    Returns:
        dict: A dictionary containing the following keys:

            - mean_db (float): Mean intensity in dB.
            - std_db (float): Standard deviation in dB.
            - range_db_ratio (float): Intensity range, expressed as a ratio in dB.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_intensity_descriptors(snd, 75, 0.01)
        {'mean_db': 70.5, 'std_db': 0.5, 'range_db_ratio': 2.5}
        ```

    Notes:
        - Hyperparameters: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Intensity___.html
        - For notes on extracting mean settings: https://www.fon.hum.uva.nl/praat/manual/Intro_6_2__Configuring_the_intensity_contour.html
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract intensity object
        intensity = snd.to_intensity(minimum_pitch=floor, time_step=frame_shift, subtract_mean=True)
        # Hyperparameters: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Intensity___.html

        # Extract descriptors
        mean_db = parselmouth.praat.call(
            intensity, "Get mean", 0, 0, "energy"
        )  # get mean - time range, time range, averaging method
        std_db = parselmouth.praat.call(intensity, "Get standard deviation", 0, 0)
        min_dB = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "parabolic")  # time range, Interpolation
        max_dB = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "parabolic")  # time range, Interpolation
        range_db_ratio = max_dB / min_dB

        # Return results
        return {"mean_db": mean_db, "std_db": std_db, "range_db_ratio": range_db_ratio}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"mean_db": np.nan, "std_db": np.nan, "range_db_ratio": np.nan}


def extract_harmonicity_descriptors(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, frame_shift: float
) -> Dict[str, float]:
    """Voice Quality - HNR.

    Function to calculate the Harmonic to Noise Ratio (HNR) in dB from a given sound object.
    This function uses the CC method as recommended by Praat.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.

    Returns:
        dict: A dictionary containing the following key:

            - hnr_db_mean (float): Mean Harmonic to Noise Ratio in dB.
            - hnr_db_std_dev (float): Harmonic to Noise Ratio standard deviation in dB.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_harmonicity_descriptors(snd, 75, 0.01)
        {'hnr_db_mean': 15.3, 'hnr_db_std_dev': 0.5}
        ```

    Notes:
        - Praat recommends using the CC method: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__cc____.html
        - Default settings can be found at: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__ac____.html
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract HNR information
        harmonicity = snd.to_harmonicity_cc(
            time_step=frame_shift, minimum_pitch=floor, silence_threshold=0.1, periods_per_window=4.5
        )
        # Praat recommends using the CC method here: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__cc____.html
        hnr_db_mean = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        hnr_db_std_dev = parselmouth.praat.call(harmonicity, "Get standard deviation", 0, 0)

        return {"hnr_db_mean": hnr_db_mean, "hnr_db_std_dev": hnr_db_std_dev}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")

        return {"hnr_db_mean": np.nan, "hnr_db_std_dev": np.nan}


def extract_slope_tilt(snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float) -> Dict[str, float]:
    """Voice Quality - Spectral Slope/Tilt.

    Function to extract spectral slope and tilt from a given sound object. This function is based on default
    Praat code adapted to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.

    Returns:
        dict: A dictionary containing the following keys:

            - spectral_slope (float): Mean spectral slope.
            - spectral_tilt (float): Mean spectral tilt.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_slope_tilt(snd, 75, 500)
        {'spectral_slope': -0.8, 'spectral_tilt': -2.5}
        ```

    Notes:
        - Spectral Slope: Ratio of energy in a spectra between 10-1000Hz over 1000-4000Hz.
        - Spectral Tilt: Linear slope of energy distribution between 100-5000Hz.
        - Using pitch-corrected LTAS to remove the effect of F0 and harmonics on the slope calculation:
        https://www.fon.hum.uva.nl/paul/papers/BoersmaKovacic2006.pdf
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        ltas_rep = parselmouth.praat.call(
            snd, "To Ltas (pitch-corrected)...", floor, ceiling, 5000, 100, 0.0001, 0.02, 1.3
        )
        # Hyperparameters: Min Pitch (Hz), Max Pitch (Hz), Maximum Frequency (Hz), Bandwidth (Hz), Shortest Period (s),
        # Longest Period (s), Maximum period factor

        spectral_slope = parselmouth.praat.call(ltas_rep, "Get slope", 50, 1000, 1000, 4000, "dB")
        # Hyperparameters: f1min, f1max, f2min, f2max, averagingUnits

        spectral_tilt_Report = parselmouth.praat.call(ltas_rep, "Report spectral tilt", 100, 5000, "Linear", "Robust")
        # Hyperparameters: minimumFrequency, maximumFrequency, Frequency Scale (linear or logarithmic),
        # Fit method (least squares or robust)

        srt_st = spectral_tilt_Report.index("Slope: ") + len("Slope: ")
        end_st = spectral_tilt_Report.index("d", srt_st)
        spectral_tilt = float(spectral_tilt_Report[srt_st:end_st])

        # Return results
        return {"spectral_slope": spectral_slope, "spectral_tilt": spectral_tilt}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"spectral_slope": np.nan, "spectral_tilt": np.nan}


def _gaussian_window(length: int) -> np.ndarray:
    """Praat's Gaussian analysis window.

    Args:
        length (int): Window length in samples.

    Returns:
        np.ndarray: The window, normalised so its edges sit at zero.
    """
    edge = np.exp(-12.0)
    phase = (np.arange(1, length + 1) - 0.5 * (length + 1)) / length
    return (np.exp(-48.0 * phase * phase) - edge) / (1.0 - edge)


def _box_average(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Centred moving average, normalised by the taps that exist rather than padded.

    Args:
        values (np.ndarray): The array to smooth.
        width (int): Window width in samples; ``<= 1`` returns the input unchanged.
        axis (int): Axis to smooth along.

    Returns:
        np.ndarray: The smoothed array, same shape as the input.
    """
    if width <= 1:
        return values
    moved = np.moveaxis(values, axis, -1)
    length = moved.shape[-1]
    low = np.clip(np.arange(length) - (width - 1) // 2, 0, length)
    high = np.clip(low + width, 0, length)
    cumulative = np.concatenate([np.zeros(moved.shape[:-1] + (1,)), np.cumsum(moved, axis=-1)], axis=-1)
    averaged = (cumulative[..., high] - cumulative[..., low]) / (high - low)
    return np.moveaxis(averaged, -1, axis)


def _robust_line_fit(x: np.ndarray, y: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one straight line per row of ``y`` by iteratively reweighted least squares.

    The weights are Huber's, with tuning constant :data:`CPPS_HUBER_K` and a median-absolute-
    deviation scale. The first pass is unweighted, so a row with no outliers is the least-squares
    line.

    Args:
        x (np.ndarray): The abscissa, shared by every row, shape ``(n,)``.
        y (np.ndarray): The ordinates, shape ``(rows, n)``.
        tolerance (float): Relative slope change below which every row has converged.

    Returns:
        tuple: The per-row slope and intercept, each shape ``(rows,)``.
    """
    weights = np.ones_like(y)
    slope = np.zeros(y.shape[0])
    intercept = np.zeros(y.shape[0])
    for _ in range(CPPS_MAX_ROBUST_ITERATIONS):
        sum_w = weights.sum(axis=1)
        sum_x = (weights * x).sum(axis=1)
        sum_y = (weights * y).sum(axis=1)
        sum_xx = (weights * x * x).sum(axis=1)
        sum_xy = (weights * x * y).sum(axis=1)
        determinant = sum_w * sum_xx - sum_x * sum_x
        determinant = np.where(determinant == 0.0, np.nan, determinant)
        next_slope = (sum_w * sum_xy - sum_x * sum_y) / determinant
        next_intercept = (sum_y - next_slope * sum_x) / sum_w
        converged = np.abs(next_slope - slope) <= tolerance * np.abs(next_slope)
        slope, intercept = next_slope, next_intercept
        if np.all(converged | ~np.isfinite(slope)):
            break
        residual = y - (intercept[:, None] + slope[:, None] * x)
        deviation = np.abs(residual - np.median(residual, axis=1, keepdims=True))
        scale = 1.4826 * np.median(deviation, axis=1, keepdims=True)
        scale = np.where(scale > 0.0, scale, 1.0)
        standardised = np.abs(residual) / (CPPS_HUBER_K * scale)
        weights = np.where(standardised <= 1.0, 1.0, 1.0 / np.where(standardised > 0.0, standardised, 1.0))
    return slope, intercept


def _smoothed_power_cepstrogram(snd: parselmouth.Sound, settings: CppsSettings) -> Tuple[np.ndarray, float]:
    """Frame the sound and return its smoothed power cepstrogram in dB.

    Args:
        snd (parselmouth.Sound): The sound to analyse.
        settings (CppsSettings): The settings the cepstrogram is computed under.

    Returns:
        tuple: The cepstrogram, shape ``(frames, quefrency bins)``, and the quefrency step in
        seconds. The cepstrogram is empty when the sound is shorter than one analysis window.
    """
    sampling_rate = 2.0 * settings.max_frequency_hz
    resampled = snd.resample(new_frequency=sampling_rate, precision=50)
    resampled.pre_emphasize(from_frequency=settings.preemphasis_from_hz)
    samples = np.asarray(resampled.values[0], dtype=float)
    samples = samples - samples.mean()

    window_s = CPPS_GAUSSIAN_WIDTH_FACTOR * CPPS_WINDOW_PERIODS / settings.pitch_floor_hz
    frame_length = 2 * int(round(window_s * sampling_rate / 2.0))
    hop = max(1, int(round(settings.time_step_s * sampling_rate)))
    if frame_length < 2 or samples.size < frame_length:
        return np.zeros((0, 0)), 1.0 / sampling_rate

    n_frames = 1 + (samples.size - frame_length) // hop
    n_fft = 1 << (frame_length - 1).bit_length()
    starts = np.arange(n_frames)[:, None] * hop + np.arange(frame_length)[None, :]
    frames = samples[starts] * _gaussian_window(frame_length)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    log_power = np.log(np.maximum(power, CPPS_QUEFRENCY_FLOOR))
    cepstrum = np.fft.irfft(log_power, n=n_fft, axis=1)[:, : n_fft // 2 + 1] ** 2

    across_time = max(1, int(round(settings.time_averaging_s / settings.time_step_s)))
    across_quefrency = max(1, int(round(settings.quefrency_averaging_s * sampling_rate)))
    smoothed = _box_average(_box_average(cepstrum, across_time, axis=0), across_quefrency, axis=1)
    return 10.0 * np.log10(np.maximum(smoothed, CPPS_QUEFRENCY_FLOOR)), 1.0 / sampling_rate


def extract_cpp_descriptors(
    snd: Union[parselmouth.Sound, Path, Audio],
    settings: CppsSettings = DEFAULT_CPPS_SETTINGS,
) -> Dict[str, float]:
    """Extract smoothed Cepstral Peak Prominence (CPPS), frame by frame.

    Each frame of the sound is taken to a log-power spectrum, inverse-transformed to a power
    cepstrum, smoothed across time and then across quefrency, and converted to dB; a straight
    trend line is fitted robustly over ``settings.trend_start_s`` to ``settings.trend_end_s``, and
    the frame's prominence is the height of the largest peak inside the quefrency band
    ``[1 / peak_search_ceiling_hz, 1 / peak_search_floor_hz]`` above that line. The three scalars
    pool every frame of the recording: no voicing gate selects frames, and no value is dropped.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        settings (CppsSettings): Every setting the measure is computed under. In triage, build it
            from the ``praat_features.cpps`` config section.

    Returns:
        dict: A dictionary containing the following keys:

            - mean_cpp (float): Mean smoothed Cepstral Peak Prominence over the frames, in dB.
            - std_dev_cpp (float): Standard deviation of that prominence across frames, in dB.
            - cpp_frames (float): Frames the two scalars rest on; 0.0 when none were placed.

        ``mean_cpp`` and ``std_dev_cpp`` are NaN when ``cpp_frames`` is 0.0 — the recording is
        shorter than one analysis window, or the analysis itself raised.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
        ValueError: If ``settings`` asks for a tilt line, an interpolation or a tilt-subtraction
            order this function does not implement.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_cpp_descriptors(snd)
        {'mean_cpp': 20.3, 'std_dev_cpp': 0.5, 'cpp_frames': 2411.0}
        ```

    Notes:
        - Cepstral Peak Prominence: the height of the cepstral peak relative to a regression line
          through the cepstrum. The *S* is the pair of smoothing windows.
        - The window is Gaussian, its effective width ``CPPS_WINDOW_PERIODS`` periods of
          ``settings.pitch_floor_hz`` and its physical duration ``CPPS_GAUSSIAN_WIDTH_FACTOR``
          times that, following Praat. The robust fit is a Huber M-estimator, which is not
          Praat's own, so values are close to but not identical with ``Get CPPS...``.
        - Every setting's derivation is in
          ``specs/20260817-triage-workflow-dag/config-derivations.md`` under ``praat_features``,
          and the rule's own record is in ``praat-instrument-audit.md`` under step 4.
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )
    if settings.tilt_line_type != "straight":
        raise ValueError(f"only a straight tilt line is implemented, not {settings.tilt_line_type!r}")
    if settings.peak_interpolation != "parabolic":
        raise ValueError(f"only parabolic peak interpolation is implemented, not {settings.peak_interpolation!r}")
    if settings.subtract_tilt_before_smoothing:
        raise ValueError("subtracting the tilt before smoothing is not implemented")

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        decibels, quefrency_step = _smoothed_power_cepstrogram(snd, settings)
        if decibels.size == 0:
            return {"mean_cpp": np.nan, "std_dev_cpp": np.nan, "cpp_frames": 0.0}

        quefrency = np.arange(decibels.shape[1]) * quefrency_step
        trend_end = settings.trend_end_s if settings.trend_end_s > 0.0 else quefrency[-1]
        fitted = np.flatnonzero((quefrency >= settings.trend_start_s) & (quefrency <= trend_end))
        searched = np.flatnonzero(
            (quefrency >= 1.0 / settings.peak_search_ceiling_hz) & (quefrency <= 1.0 / settings.peak_search_floor_hz)
        )
        if fitted.size < 2 or searched.size == 0:
            return {"mean_cpp": np.nan, "std_dev_cpp": np.nan, "cpp_frames": 0.0}

        rows = np.arange(decibels.shape[0])
        peak = searched[decibels[:, searched].argmax(axis=1)]
        before = decibels[rows, np.maximum(peak - 1, 0)]
        at = decibels[rows, peak]
        after = decibels[rows, np.minimum(peak + 1, decibels.shape[1] - 1)]
        curvature = before - 2.0 * at + after
        offset = np.clip(
            np.where(curvature < 0.0, 0.5 * (before - after) / np.where(curvature < 0.0, curvature, 1.0), 0.0),
            -0.5,
            0.5,
        )
        peak_db = at - 0.25 * (before - after) * offset
        peak_quefrency = (peak + offset) * quefrency_step

        slope, intercept = _robust_line_fit(quefrency[fitted], decibels[:, fitted], settings.robust_tolerance)
        prominence = peak_db - (intercept + slope * peak_quefrency)

        finite = prominence[np.isfinite(prominence)]
        if finite.size == 0:
            return {"mean_cpp": np.nan, "std_dev_cpp": np.nan, "cpp_frames": 0.0}
        return {
            "mean_cpp": float(np.mean(finite)),
            "std_dev_cpp": float(np.std(finite)),
            "cpp_frames": float(finite.size),
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"mean_cpp": np.nan, "std_dev_cpp": np.nan, "cpp_frames": 0.0}


def measure_f1f2_formants_bandwidths(
    snd: Union[parselmouth.Sound, Path, Audio],
    floor: float,
    ceiling: float,
    frame_shift: float,
    max_formants: int = 5,
    maximum_formant_hz: float = 5000.0,
    window_length: float = 0.025,
    pre_emphasis_from_hz: float = 50.0,
) -> Dict[str, float]:
    """Extract Formant Frequency Features.

    Function to extract formant frequency features from a given sound object. This function is adapted from default
    Praat code to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.
        max_formants (int, optional): Maximum number of formants to measure. Defaults to 5.
        maximum_formant_hz (float, optional): Maximum formant frequency to measure. Defaults to 5000.0.
        window_length (float, optional): Window length for formant analysis. Defaults to 0.025.
        pre_emphasis_from_hz (float, optional): Pre-emphasis frequency for formant analysis. Defaults to 50.0.

    Returns:
        dict: A dictionary containing the following keys:

            - f1_mean (float): Mean F1 location.
            - f1_std (float): Standard deviation of F1 location.
            - b1_mean (float): Mean F1 bandwidth.
            - b1_std (float): Standard deviation of F1 bandwidth.
            - f2_mean (float): Mean F2 location.
            - f2_std (float): Standard deviation of F2 location.
            - b2_mean (float): Mean F2 bandwidth.
            - b2_std (float): Standard deviation of F2 bandwidth.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> measureFormants(snd, 75, 500, 0.01)
        {'f1_mean': 500.0, 'f1_std': 50.0, 'b1_mean': 80.0, 'b1_std': 10.0, 'f2_mean': 1500.0,
        'f2_std': 100.0, 'b2_mean': 120.0, 'b2_std': 20.0}
        ```

    Notes:
        - Formants are the resonances of the vocal tract, determined by tongue placement and vocal tract shape.
        - Mean F1 typically varies between 300 to 750 Hz, while mean F2 typically varies between 900 to 2300 Hz.
        - Formant bandwidth is measured by taking the width of the band forming 3 dB down from the formant peak.
        - Formant extraction occurs per pitch period (pulses), meaning that the analysis identifies the points in the
          sound where the vocal folds come together, helping to align the formant measurements precisely with the
          pitch periods.
        - Adapted from code at this [link](https://osf.io/6dwr3/).
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError("`parselmouth` is not installed. Install with `pip install senselab`.")
    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        formants = snd.to_formant_burg(
            time_step=frame_shift,
            max_number_of_formants=max_formants,
            maximum_formant=maximum_formant_hz,
            window_length=window_length,
            pre_emphasis_from=pre_emphasis_from_hz,
        )
        # Key Hyperparameters: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html

        pitch = snd.to_pitch_cc(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

        n = parselmouth.praat.call(pulses, "Get number of points")
        if n == 0:
            return {
                k: float("nan")
                for k in ("f1_mean", "f1_std", "b1_mean", "b1_std", "f2_mean", "f2_std", "b2_mean", "b2_std")
            }

        times = np.array(
            [parselmouth.praat.call(pulses, "Get time from index", i + 1) for i in range(n)],
            dtype=float,
        )

        # Sample at those times (native calls)
        f1 = np.array(
            [formants.get_value_at_time(1, t, unit=parselmouth.FormantUnit.HERTZ) for t in times], dtype=float
        )
        b1 = np.array(
            [formants.get_bandwidth_at_time(1, t, unit=parselmouth.FormantUnit.HERTZ) for t in times], dtype=float
        )
        f2 = np.array(
            [formants.get_value_at_time(2, t, unit=parselmouth.FormantUnit.HERTZ) for t in times], dtype=float
        )
        b2 = np.array(
            [formants.get_bandwidth_at_time(2, t, unit=parselmouth.FormantUnit.HERTZ) for t in times], dtype=float
        )

        return {
            "f1_mean": float(np.nanmean(f1)),
            "f1_std": float(np.nanstd(f1)),
            "b1_mean": float(np.nanmean(b1)),
            "b1_std": float(np.nanstd(b1)),
            "f2_mean": float(np.nanmean(f2)),
            "f2_std": float(np.nanstd(f2)),
            "b2_mean": float(np.nanmean(b2)),
            "b2_std": float(np.nanstd(b2)),
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            k: float("nan")
            for k in ("f1_mean", "f1_std", "b1_mean", "b1_std", "f2_mean", "f2_std", "b2_mean", "b2_std")
        }


def extract_spectral_moments(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float, window_size: float, frame_shift: float
) -> Dict[str, float]:
    """Extract Spectral Moments.

    Function to extract spectral moments from a given sound object. This function is adapted from default
    Praat code to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        window_size (float): Time frame over which the spectra is calculated, typically set to 25 ms.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.

    Returns:
        dict: A dictionary containing the following keys:

            - spectral_gravity (float): Mean spectral gravity.
            - spectral_std_dev (float): Mean spectral standard deviation.
            - spectral_skewness (float): Mean spectral skewness.
            - spectral_kurtosis (float): Mean spectral kurtosis.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_spectral_moments(snd, 75, 500, 0.025, 0.01)
        {'spectral_gravity': 5000.0, 'spectral_std_dev': 150.0, 'spectral_skewness': -0.5, 'spectral_kurtosis': 3.0}
        ```

    Notes:
        - Spectral Gravity: Measure for how high the frequencies in a spectrum are on average over the entire frequency
        domain weighted by the power spectrum.
        - Spectral Standard Deviation: Measure for how much the frequencies in a spectrum can deviate from the centre
        of gravity.
        - Spectral Skewness: Measure for how much the shape of the spectrum below the centre of gravity is different
        from the shape above the mean frequency.
        - Spectral Kurtosis: Measure for how much the shape of the spectrum around the centre of gravity is different
          from a Gaussian shape.
        - Details: https://www.fon.hum.uva.nl/praat/manual/Spectrum__Get_central_moment___.html
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract pitch object for voiced checking
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)

        # Calculate Spectrogram
        spectrogram = snd.to_spectrogram(window_length=window_size, time_step=frame_shift)
        # Using default settings other than window length and frame shift
        # Details: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Spectrogram___.html

        Gravity_list, STD_list, Skew_list, Kurt_list = [], [], [], []

        num_steps = spectrogram.nx  # Number of frames exposed in the spectrogram

        for i in range(1, num_steps + 1):
            t = spectrogram.x1 + (i - 1) * spectrogram.dx
            # where x1 is the time of the center of the first frame
            # and dx is the time step (seconds between frames)
            # This is equivalent as doing
            # t = parselmouth.praat.call(spectrogram, "Get time from frame number", i)

            pitch_value = pitch.get_value_at_time(t)

            if not np.isnan(pitch_value):
                voiced_spectrum = spectrogram.to_spectrum_slice(t)
                # Details: https://www.fon.hum.uva.nl/praat/manual/Spectrogram__To_Spectrum__slice____.html

                Gravity_LLD = voiced_spectrum.get_centre_of_gravity(power=2)
                if not np.isnan(Gravity_LLD):
                    Gravity_list.append(Gravity_LLD)

                STD_LLD = voiced_spectrum.get_standard_deviation(power=2)
                if not np.isnan(STD_LLD):
                    STD_list.append(STD_LLD)

                Skew_LLD = voiced_spectrum.get_skewness(power=2)
                if not np.isnan(Skew_LLD):
                    Skew_list.append(Skew_LLD)

                Kurt_LLD = voiced_spectrum.get_kurtosis(power=2)
                if not np.isnan(Kurt_LLD):
                    Kurt_list.append(Kurt_LLD)

        gravity_mean = np.mean(Gravity_list) if Gravity_list else np.nan
        std_mean = np.mean(STD_list) if STD_list else np.nan
        skew_mean = np.mean(Skew_list) if Skew_list else np.nan
        kurt_mean = np.mean(Kurt_list) if Kurt_list else np.nan

        return {
            "spectral_gravity": gravity_mean,
            "spectral_std_dev": std_mean,
            "spectral_skewness": skew_mean,
            "spectral_kurtosis": kurt_mean,
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "spectral_gravity": np.nan,
            "spectral_std_dev": np.nan,
            "spectral_skewness": np.nan,
            "spectral_kurtosis": np.nan,
        }


### More functions ###


def extract_audio_duration(snd: Union[parselmouth.Sound, Path, Audio]) -> Dict[str, float]:
    """Get the duration of a given audio file or Audio object.

    This function calculates the total duration of an audio file or audio object
    by creating a Parselmouth `Sound` object and then calling a Praat method
    to retrieve the duration of the audio in seconds.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object,
        a file path (Path), or an `Audio` object containing the audio waveform and
        its corresponding sampling rate.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "duration" (float): The total duration of the audio in seconds.

    Raises:
        FileNotFoundError: If a provided file path does not exist.

    Example:
        ```python
        >>> snd = Audio(waveform=[...], sampling_rate=16000)
        >>> extract_audio_duration(snd)
        {'duration': 5.23}
        ```
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    # Check if the input is a Path, in which case we load the audio from the file
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

    try:
        # Return the duration in a dictionary
        return {"duration": snd.duration}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"duration": np.nan}


def extract_jitter(snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float) -> Dict[str, float]:
    """Returns the jitter descriptors for the given sound or audio file.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object, a file path (Path),
        or an `Audio` object containing the audio waveform and its corresponding sampling rate.
        floor (float): Minimum fundamental frequency (F0) in Hz.
        ceiling (float): Maximum fundamental frequency (F0) in Hz.

    Returns:
        Dict[str, float]: A dictionary containing various jitter measurements.
    """

    def _to_point_process(sound: parselmouth.Sound, f0min: float, f0max: float) -> parselmouth.Data:
        return parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    def _extract_jitter(type: str, point_process: parselmouth.Data) -> float:
        return parselmouth.praat.call(point_process, f"Get jitter ({type})", 0, 0, 0.0001, 0.02, 1.3)

    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    # Check if the input is a Path or Audio, and convert to Parselmouth Sound if necessary
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

    try:
        # Convert the sound to a point process for jitter measurement
        point_process = _to_point_process(snd, floor, ceiling)

        # Extract jitter measures from the point process
        return {
            "local_jitter": _extract_jitter("local", point_process),
            "localabsolute_jitter": _extract_jitter("local, absolute", point_process),
            "rap_jitter": _extract_jitter("rap", point_process),
            "ppq5_jitter": _extract_jitter("ppq5", point_process),
            "ddp_jitter": _extract_jitter("ddp", point_process),
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "local_jitter": np.nan,
            "localabsolute_jitter": np.nan,
            "rap_jitter": np.nan,
            "ppq5_jitter": np.nan,
            "ddp_jitter": np.nan,
        }


def extract_shimmer(snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float) -> Dict[str, float]:
    """Returns the shimmer descriptors for the given sound or audio file.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object, a file path (Path),
        or an `Audio` object containing the audio waveform and its corresponding sampling rate.
        floor (float): Minimum fundamental frequency (F0) in Hz.
        ceiling (float): Maximum fundamental frequency (F0) in Hz.

    Returns:
        Dict[str, float]: A dictionary containing various shimmer measurements.
    """

    def _to_point_process(sound: parselmouth.Sound, f0min: float, f0max: float) -> parselmouth.Data:
        return parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    def _extract_shimmer(type: str, sound: parselmouth.Sound, point_process: parselmouth.Data) -> float:
        return parselmouth.praat.call([sound, point_process], f"Get shimmer ({type})", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    # Check if the input is a Path or Audio, and convert to Parselmouth Sound if necessary
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

    try:
        # Convert the sound to a point process for shimmer measurement
        point_process = _to_point_process(snd, floor, ceiling)

        # Extract shimmer measures from the sound and point process
        return {
            "local_shimmer": _extract_shimmer("local", snd, point_process),
            "localDB_shimmer": _extract_shimmer("local_dB", snd, point_process),
            "apq3_shimmer": _extract_shimmer("apq3", snd, point_process),
            "apq5_shimmer": _extract_shimmer("apq5", snd, point_process),
            "apq11_shimmer": _extract_shimmer("apq11", snd, point_process),
            "dda_shimmer": _extract_shimmer("dda", snd, point_process),
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "local_shimmer": np.nan,
            "localDB_shimmer": np.nan,
            "apq3_shimmer": np.nan,
            "apq5_shimmer": np.nan,
            "apq11_shimmer": np.nan,
            "dda_shimmer": np.nan,
        }


# Wrapper
def extract_praat_parselmouth_features_from_audios(
    audios: List[Audio],
    time_step: float = 0.005,
    window_length: float = 0.025,
    pitch_unit: str = "Hertz",
    search_floor_hz: float = 50.0,
    search_ceiling_hz: float = 600.0,
    pitch_floor_divisor: float = DEFAULT_PITCH_FLOOR_DIVISOR,
    pitch_ceiling_quartile_multiplier: float = DEFAULT_PITCH_CEILING_QUARTILE_MULTIPLIER,
    pitch_pinned_percentile: float = DEFAULT_PITCH_PINNED_PERCENTILE,
    pitch_excursion_multiplier: float = DEFAULT_PITCH_EXCURSION_MULTIPLIER,
    pitch_pinned_octave_ratio: float = DEFAULT_PITCH_PINNED_OCTAVE_RATIO,
    cpps: CppsSettings = DEFAULT_CPPS_SETTINGS,
    speech_rate: bool = True,
    intensity_descriptors: bool = True,
    harmonicity_descriptors: bool = True,
    formants: bool = True,
    spectral_moments: bool = True,
    pitch: bool = True,
    slope_tilt: bool = True,
    cpp_descriptors: bool = True,
    duration: bool = True,
    jitter: bool = True,
    shimmer: bool = True,
    n_jobs: int = 1,
    backend: Literal["threading", "loky", "multiprocessing", "sequential"] = "sequential",
    verbose: int = 0,
    cache_dir: Optional[str | os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Extract Praat/Parselmouth features per `Audio`.

    Parallelizes **across audios** and optionally caches per-audio computations.
    Toggle individual feature blocks with the boolean flags.

    Args:
        audios (list): List of Audio objects to extract features from.
        time_step (float): Time rate at which to extract features. Defaults to 0.005.
        window_length (float): Window length in seconds for spectral features. Defaults to 0.025.
        pitch_unit (str): Unit for pitch measurements. Defaults to "Hertz".
        search_floor_hz (float): Lowest pitch of the wide search each recording's range is narrowed from.
        search_ceiling_hz (float): Highest pitch of that wide search.
        pitch_floor_divisor (float): Forwarded to :func:`extract_pitch_values`.
        pitch_ceiling_quartile_multiplier (float): Forwarded to :func:`extract_pitch_values`.
        pitch_pinned_percentile (float): Forwarded to :func:`extract_pitch_values`.
        pitch_excursion_multiplier (float): Forwarded to :func:`extract_pitch_values`.
        pitch_pinned_octave_ratio (float): Forwarded to :func:`extract_pitch_values`.
        cpps (CppsSettings): Forwarded to :func:`extract_cpp_descriptors`.
        speech_rate (bool): Whether to extract speech rate. Defaults to True.
        intensity_descriptors (bool): Whether to extract intensity descriptors. Defaults to True.
        harmonicity_descriptors (bool): Whether to extract harmonic descriptors. Defaults to True.
        formants (bool): Whether to extract formants. Defaults to True.
        spectral_moments (bool): Whether to extract spectral moments. Defaults to True.
        pitch (bool): Whether to extract pitch. Defaults to True.
        slope_tilt (bool): Whether to extract slope and tilt. Defaults to True.
        cpp_descriptors (bool): Whether to extract CPP descriptors. Defaults to True.
        duration (bool): Whether to extract duration. Defaults to True.
        jitter (bool): Whether to extract jitter. Defaults to True.
        shimmer (bool): Whether to extract shimmer. Defaults to True.
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
        list[dict[str, Any]]: A list of JSON-like dictionaries with extracted features
            structured under "praat_parselmouth". Each carries the five keys
            :func:`extract_pitch_values` returned — ``pitch_floor``, ``pitch_ceiling``,
            ``pitch_frames``, ``pitch_failed`` and ``pitch_range_fell_back`` — under those
            names, alongside the scalars they conditioned.

    """

    # Utility function to extract features per-audio worker
    def _extract_one(snd: Audio) -> Dict[str, Any]:
        # Shared precomputations
        pitch_values_out = extract_pitch_values(
            snd=snd,
            search_floor_hz=search_floor_hz,
            search_ceiling_hz=search_ceiling_hz,
            pitch_floor_divisor=pitch_floor_divisor,
            pitch_ceiling_quartile_multiplier=pitch_ceiling_quartile_multiplier,
            pitch_pinned_percentile=pitch_pinned_percentile,
            pitch_excursion_multiplier=pitch_excursion_multiplier,
            pitch_pinned_octave_ratio=pitch_pinned_octave_ratio,
        )
        pitch_floor = pitch_values_out["pitch_floor"]
        pitch_ceiling = pitch_values_out["pitch_ceiling"]

        # Conditionally compute blocks
        speech_rate_out = extract_speech_rate(snd=snd) if speech_rate else None

        pitch_out = (
            extract_pitch_descriptors(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
                frame_shift=time_step,
                unit=pitch_unit,
            )
            if pitch
            else None
        )

        intensity_out = (
            extract_intensity_descriptors(
                snd=snd,
                floor=pitch_floor,
                frame_shift=time_step,
            )
            if intensity_descriptors
            else None
        )

        harmonicity_out = (
            extract_harmonicity_descriptors(
                snd=snd,
                floor=pitch_floor,
                frame_shift=time_step,
            )
            if harmonicity_descriptors
            else None
        )

        formants_out = (
            measure_f1f2_formants_bandwidths(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
                frame_shift=time_step,
            )
            if formants
            else None
        )

        spectral_moments_out = (
            extract_spectral_moments(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
                window_size=window_length,
                frame_shift=time_step,
            )
            if spectral_moments
            else None
        )

        slope_tilt_out = (
            extract_slope_tilt(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
            )
            if slope_tilt
            else None
        )

        cpp_out = extract_cpp_descriptors(snd=snd, settings=cpps) if cpp_descriptors else None

        audio_duration_out = extract_audio_duration(snd=snd) if duration else None

        jitter_out = (
            extract_jitter(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
            )
            if jitter
            else None
        )

        shimmer_out = (
            extract_shimmer(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
            )
            if shimmer
            else None
        )

        # collect outputs
        unit_l = pitch_unit.lower()
        feature_data: Dict[str, Any] = dict(pitch_values_out)

        if duration and audio_duration_out is not None:
            feature_data["duration"] = audio_duration_out["duration"]

        if speech_rate and speech_rate_out is not None:
            feature_data["speaking_rate"] = speech_rate_out["speaking_rate"]
            feature_data["articulation_rate"] = speech_rate_out["articulation_rate"]
            feature_data["phonation_ratio"] = speech_rate_out["phonation_ratio"]
            feature_data["pause_rate"] = speech_rate_out["pause_rate"]
            feature_data["mean_pause_duration"] = speech_rate_out["mean_pause_dur"]

        if pitch and pitch_out is not None:
            feature_data[f"mean_f0_{unit_l}"] = pitch_out[f"mean_f0_{unit_l}"]
            feature_data[f"std_f0_{unit_l}"] = pitch_out[f"stdev_f0_{unit_l}"]

        if intensity_descriptors and intensity_out is not None:
            feature_data["mean_intensity_db"] = intensity_out["mean_db"]
            feature_data["std_intensity_db"] = intensity_out["std_db"]
            feature_data["range_ratio_intensity_db"] = intensity_out["range_db_ratio"]

        if harmonicity_descriptors and harmonicity_out is not None:
            feature_data["mean_hnr_db"] = harmonicity_out["hnr_db_mean"]
            feature_data["std_hnr_db"] = harmonicity_out["hnr_db_std_dev"]

        if slope_tilt and slope_tilt_out is not None:
            feature_data["spectral_slope"] = slope_tilt_out["spectral_slope"]
            feature_data["spectral_tilt"] = slope_tilt_out["spectral_tilt"]

        if cpp_descriptors and cpp_out is not None:
            feature_data["cepstral_peak_prominence_mean"] = cpp_out["mean_cpp"]
            feature_data["cepstral_peak_prominence_std"] = cpp_out["std_dev_cpp"]
            feature_data["cepstral_peak_prominence_frames"] = cpp_out["cpp_frames"]

        if formants and formants_out is not None:
            feature_data["mean_f1_loc"] = formants_out["f1_mean"]
            feature_data["std_f1_loc"] = formants_out["f1_std"]
            feature_data["mean_b1_loc"] = formants_out["b1_mean"]
            feature_data["std_b1_loc"] = formants_out["b1_std"]
            feature_data["mean_f2_loc"] = formants_out["f2_mean"]
            feature_data["std_f2_loc"] = formants_out["f2_std"]
            feature_data["mean_b2_loc"] = formants_out["b2_mean"]
            feature_data["std_b2_loc"] = formants_out["b2_std"]

        if spectral_moments and spectral_moments_out is not None:
            feature_data["spectral_gravity"] = spectral_moments_out["spectral_gravity"]
            feature_data["spectral_std_dev"] = spectral_moments_out["spectral_std_dev"]
            feature_data["spectral_skewness"] = spectral_moments_out["spectral_skewness"]
            feature_data["spectral_kurtosis"] = spectral_moments_out["spectral_kurtosis"]

        if jitter and jitter_out is not None:
            feature_data["local_jitter"] = jitter_out["local_jitter"]
            feature_data["localabsolute_jitter"] = jitter_out["localabsolute_jitter"]
            feature_data["rap_jitter"] = jitter_out["rap_jitter"]
            feature_data["ppq5_jitter"] = jitter_out["ppq5_jitter"]
            feature_data["ddp_jitter"] = jitter_out["ddp_jitter"]

        if shimmer and shimmer_out is not None:
            feature_data["local_shimmer"] = shimmer_out["local_shimmer"]
            feature_data["localDB_shimmer"] = shimmer_out["localDB_shimmer"]
            feature_data["apq3_shimmer"] = shimmer_out["apq3_shimmer"]
            feature_data["apq5_shimmer"] = shimmer_out["apq5_shimmer"]
            feature_data["apq11_shimmer"] = shimmer_out["apq11_shimmer"]
            feature_data["dda_shimmer"] = shimmer_out["dda_shimmer"]

        return feature_data

    # optional cache
    memory: Optional[Memory] = Memory(str(cache_dir), verbose=verbose) if cache_dir else None
    if memory:
        _extract_one = memory.cache(_extract_one)

    # parallel across audios
    return Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
    )(delayed(_extract_one)(a) for a in audios)
