"""This module contains functions that extract features from audio files using the PRAAT library.

The initial implementation of this features extraction was started by Nicholas Cummins
from King's College London and has since been further developed and maintained
by the senselab community.
"""

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
        return {
            "speaking_rate": np.nan,
            "articulation_rate": np.nan,
            "phonation_ratio": np.nan,
            "pause_rate": np.nan,
            "mean_pause_dur": np.nan,
        }


def extract_pitch_values(snd: Union[parselmouth.Sound, Path, Audio]) -> Dict[str, float]:
    """Estimate Pitch Range.

    Calculates the mean pitch using a wide range and uses this to shorten the range for future pitch extraction
    algorithms.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.

    Returns:
        dict: A dictionary containing the following keys:

            - pitch_floor (float): The lowest pitch value to use in future pitch extraction algorithms.
            - pitch_ceiling (float): The highest pitch value to use in future pitch extraction algorithms.

    Notes:
        Values are taken from: [Standardization of pitch-range settings in voice acoustic analysis](https://doi.org/10.3758/BRM.41.2.318)

        The problem observed with doing a really broad pitch search was the occasional error if F1 was low.
        So crude outlier detection is used to help with this.

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
        >>> pitch_values(snd)
        {'pitch_floor': 60, 'pitch_ceiling': 250}
        ```
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        pitch_wide = snd.to_pitch_ac(time_step=0.005, pitch_floor=50, pitch_ceiling=600)
        # Other than values above, I'm using default hyperparamters
        # Details: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html

        # remove outliers from wide pitch search
        pitch_values = pitch_wide.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values != 0]
        pitch_values_Z = (pitch_values - np.mean(pitch_values)) / np.std(pitch_values)
        pitch_values_filtered = pitch_values[abs(pitch_values_Z) <= 2]

        mean_pitch = np.mean(pitch_values_filtered)

        # Here there is an interesting alternative solution to discuss: https://praatscripting.lingphon.net/conditionals-1.html
        if mean_pitch < 170:
            # 'male' settings
            pitch_floor = 60.0
            pitch_ceiling = 250.0
        else:
            # 'female' and 'child' settings
            pitch_floor = 100.0
            pitch_ceiling = 500.0

        return {"pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"pitch_floor": np.nan, "pitch_ceiling": np.nan}


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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
        return {"spectral_slope": np.nan, "spectral_tilt": np.nan}


def extract_cpp_descriptors(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float, frame_shift: float
) -> Dict[str, float]:
    """Extract Cepstral Peak Prominence (CPP).

    Function to calculate the Cepstral Peak Prominence (CPP) from a given sound object.
    This function is adapted from default Praat code to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.

    Returns:
        dict: A dictionary containing the following key:

            - mean_cpp (float): Mean Cepstral Peak Prominence.
            - std_dev_cpp (float): Standard deviation in Cepstral Peak Prominence.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_CPP(snd, 75, 500, 0.01)
        {'mean_cpp': 20.3, 'std_dev_cpp': 0.5}
        ```

    Notes:
        - Cepstral Peak Prominence: The height (i.e., “prominence”) of that peak relative to a regression line
        through the overall cepstrum.
        - Adapted from: https://osf.io/ctwgr and http://phonetics.linguistics.ucla.edu/facilities/acoustic/voiced_extract_auto.txt
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract pitch object for voiced checking
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling, voicing_threshold=0.3)

        pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

        textgrid = parselmouth.praat.call(pulses, "To TextGrid (vuv)", 0.02, 0.1)

        vuv_table = parselmouth.praat.call(textgrid, "Down to Table", "no", 6, "yes", "no")
        # Variables - include line number, Time decimals, include tier names, include empty intervals

        cpp_list = []

        n_intervals = parselmouth.praat.call(vuv_table, "Get number of rows")
        for i in range(n_intervals):
            label = parselmouth.praat.call(vuv_table, "Get value", i + 1, "text")
            if label == "V":
                tmin = parselmouth.praat.call(vuv_table, "Get value", i + 1, "tmin")
                tmax = parselmouth.praat.call(vuv_table, "Get value", i + 1, "tmax")
                snd_segment = snd.extract_part(float(tmin), float(tmax))

                PowerCepstrogram = parselmouth.praat.call(snd_segment, "To PowerCepstrogram", 60, 0.002, 5000, 50)
                # PowerCepstrogram (60-Hz pitch floor, 2-ms time step, 5-kHz maximum frequency,
                # and pre-emphasis from 50 Hz)

                try:
                    CPP_Value = parselmouth.praat.call(
                        PowerCepstrogram,
                        "Get CPPS...",
                        "no",
                        0.01,
                        0.001,
                        60,
                        330,
                        0.05,
                        "parabolic",
                        0.001,
                        0,
                        "Straight",
                        "Robust",
                    )
                    # Subtract tilt before smoothing = “no”; time averaging window = 0.01 s;
                    # quefrency averaging window = 0.001 s;
                    # Peak search pitch range = 60–330 Hz; tolerance = 0.05; interpolation = “Parabolic”;
                    # tilt line frequency range = 0.001–0 s (no upper bound);
                    # Line type = “Straight”; fit method = “Robust.”
                except Exception as e:
                    current_frame = inspect.currentframe()
                    if current_frame is not None:
                        current_function_name = current_frame.f_code.co_name
                        logger.error(f'Error in "{current_function_name}": \n' + str(e))
                    CPP_Value = np.nan

                if not np.isnan(CPP_Value) and CPP_Value > 4:
                    cpp_list.append(CPP_Value)

        # Calculate Final Features
        if cpp_list:
            CPP_array = np.array(cpp_list)
            CPP_mean = np.mean(CPP_array)
            CPP_std = np.std(CPP_array)
        else:
            CPP_mean = np.nan
            CPP_std = np.nan

        # Return Result
        return {"mean_cpp": CPP_mean, "std_dev_cpp": CPP_std}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"mean_cpp": np.nan, "std_dev_cpp": np.nan}


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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            "`parselmouth` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
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
            structured under "praat_parselmouth".

    """

    # Utility function to extract features per-audio worker
    def _extract_one(snd: Audio) -> Dict[str, Any]:
        # Shared precomputations
        pitch_values_out = extract_pitch_values(snd=snd)
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

        cpp_out = (
            extract_cpp_descriptors(
                snd=snd,
                floor=pitch_floor,
                ceiling=pitch_ceiling,
                frame_shift=time_step,
            )
            if cpp_descriptors
            else None
        )

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
        feature_data: Dict[str, Any] = {}

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
