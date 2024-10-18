"""This module contains functions that extract features from audio files using the PRAAT library."""

import inspect
from pathlib import Path
from typing import Dict, Union

import numpy as np
import parselmouth  # type: ignore
import pydra  # type: ignore

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import logger


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
    if parselmouth.praat.call(snd_full, "Get number of channels") > 1:
        snd_full = snd_full.convert_to_mono()
    if parselmouth.praat.call(snd_full, "Get sampling frequency") != sampling_rate:
        snd_full = parselmouth.praat.call(snd_full, "Resample", sampling_rate, 50)
        # Details of queery: https://www.fon.hum.uva.nl/praat/manual/Get_sampling_frequency.html
        # Details of conversion: https://www.fon.hum.uva.nl/praat/manual/Sound__Resample___.html

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
    try:
        # _____________________________________________________________________________________________________________
        # Load the sound object into parselmouth if it is an Audio object
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # _____________________________________________________________________________________________________________
        # Key pause detection hyperparameters

        # Silence Threshold (dB) - standard setting to detect silence in the "To TextGrid (silences)" function.
        # The higher this number, the lower the chances of finding silent pauses
        silencedb = -25

        # Minimum_dip_between_peaks_(dB) - if there are decreases in intensity
        # of at least this value surrounding the peak, the peak is labelled to be a syllable nucleus
        # I.e. the size of the dip between two possible peakes
        # The higher this number, the less syllables will be found
        # For clean and filtered signal use 4, if not use 2 (recommend thresholds)
        mindip = 4
        # Code for determining if the signal not clean/filtered
        hnr = parselmouth.praat.call(
            snd.to_harmonicity_cc(), "Get mean", 0, 0
        )  # Note: (0,0) is the time range for extraction, setting both two zero tells praat to use the full file
        if hnr < 60:
            mindip = 2

        # Minimum pause duration (s): How long should a pause be to be counted as a silent pause?
        # The higher this number, the fewer pauses will be found
        minpause = 0.3  # the default for this is 0.1 in Praat, the de Jong's script has this set at 0.3
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
        silencedb_1 = max_99_intensity + silencedb
        dB_adjustment = max_intensity - max_99_intensity
        silencedb_2 = silencedb - dB_adjustment
        if silencedb_1 < min_intensity:
            silencedb_1 = min_intensity

        # ______________________________________________________________________________________________________________
        # Create a TextGrid in which the silent and sounding intervals, store these intervals

        textgrid = parselmouth.praat.call(
            intensity, "To TextGrid (silences)", silencedb_2, minpause, 0.1, "silent", "sounding"
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

        Phonation_Time = 0
        for ipause in range(npauses):
            pause = ipause + 1
            beginsound = parselmouth.praat.call(silencetable, "Get value", pause, 1)
            endsound = parselmouth.praat.call(silencetable, "Get value", pause, 2)
            speakingdur = endsound - beginsound

            Phonation_Time += speakingdur

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
            if value > silencedb_1:
                peakcount += 1
                intensities.append(value)
                timepeaks.append(t[i])

        # ______________________________________________________________________________________________________________
        # Now find all valid peaks

        # fill array with valid peaks: only intensity values if preceding
        # dip in intensity is greater than mindip
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
            if diffint > mindip:
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

        Number_Syllables = int(0)
        for time in range(validpeakcount):
            querytime = validtime[time]
            whichinterval = parselmouth.praat.call(textgrid, "Get interval at time", 1, querytime)
            whichlabel = parselmouth.praat.call(textgrid, "Get label of interval", 1, whichinterval)
            value = pitch.get_value_at_time(querytime)
            if not np.isnan(value):
                if whichlabel == "sounding":
                    Number_Syllables += 1

        # ______________________________________________________________________________________________________________
        # return results

        Original_Dur = end_speak - begin_speak

        speaking_rate = Number_Syllables / Original_Dur
        articulation_rate = Number_Syllables / Phonation_Time
        phonation_ratio = Phonation_Time / Original_Dur

        Number_Pauses = npauses - 1
        Pause_Time = Original_Dur - Phonation_Time

        pause_rate = Number_Pauses / Original_Dur
        Mean_Pause_Dur = Pause_Time / Number_Pauses if Number_Pauses > 0 else 0.0

        return {
            "speaking_rate": speaking_rate,
            "articulation_rate": articulation_rate,
            "phonation_ratio": phonation_ratio,
            "pause_rate": pause_rate,
            "mean_pause_dur": Mean_Pause_Dur,
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {
            "speaking_rate": float("nan"),
            "articulation_rate": float("nan"),
            "phonation_ratio": float("nan"),
            "pause_rate": float("nan"),
            "mean_pause_dur": float("nan"),
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

        if mean_pitch < 170:
            # 'male' settings
            pitch_floor = 60.0
            pitch_ceiling = 250.0
        else:
            # 'female' settings
            pitch_floor = 100.0
            pitch_ceiling = 500.0

        return {"pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"pitch_floor": float("nan"), "pitch_ceiling": float("nan")}


def extract_pitch(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float, frame_shift: float, unit: str = "Hertz"
) -> Dict[str, float]:
    """Extract Pitch Features.

    Function to extract key pitch features from a given sound object.
    This function uses the pitch_ac method as autocorrelation is better at finding intended intonation contours.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.
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
        >>> extract_pitch(snd, 75, 500, 0.01, "Hertz")
        {'mean_f0_hertz': 220.5, 'stdev_f0_Hertz': 2.5}
        ```
    """
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
        return {f"mean_f0_{unit.lower()}": float("nan"), f"stdev_f0_{unit.lower()}": float("nan")}


def extract_intensity(snd: Union[parselmouth.Sound, Path, Audio], floor: float, frame_shift: float) -> Dict[str, float]:
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
            - range_db_ratio (float): Intensity range, expressed as a ratio in dB.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_intensity(snd, 75, 0.01)
        {'mean_db': 70.5, 'range_db_ratio': 2.5}
        ```

    Notes:
        - Hyperparameters: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Intensity___.html
        - For notes on extracting mean settings: https://www.fon.hum.uva.nl/praat/manual/Intro_6_2__Configuring_the_intensity_contour.html
    """
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
        min_dB = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "parabolic")  # time range, Interpolation
        max_dB = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "parabolic")  # time range, Interpolation
        range_dB_Ratio = max_dB / min_dB

        # Return results
        print(
            f"mean_db: {mean_db}, range_db_ratio: {range_dB_Ratio}")
        return {"mean_db": mean_db, "range_db_ratio": range_dB_Ratio}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"mean_db": float("nan"), "range_db_ratio": float("nan")}


def extract_harmonicity(
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

            - HNR_db_mean (float): Mean Harmonic to Noise Ratio in dB.
            - HNR_db_std_dev (float): Harmonic to Noise Ratio standard deviation in dB.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_harmonicity(snd, 75, 0.01)
        {'HNR_db_mean': 15.3, 'HNR_db_std_dev': 0.5}
        ```

    Notes:
        - Praat recommends using the CC method: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__cc____.html
        - Default settings can be found at: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__ac____.html
    """
    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract HNR information
        harmonicity = snd.to_harmonicity_cc(
            time_step=frame_shift, minimum_pitch=floor, silence_threshold=0.1, periods_per_window=4.5
        )
        # Praat recommends using the CC method here: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Harmonicity__cc____.html

        HNR_db_mean = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        HNR_db_std_dev = parselmouth.praat.call(harmonicity, "Get standard deviation", 0, 0)

        return {"HNR_db_mean": HNR_db_mean, "HNR_db_std_dev": HNR_db_std_dev}
    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"HNR_db_mean": float("nan"), "HNR_db_std_dev": float("nan")}


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

            - spc_slope (float): Mean spectral slope.
            - spc_tilt (float): Mean spectral tilt.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_slope_tilt(snd, 75, 500)
        {'spc_slope': -0.8, 'spc_tilt': -2.5}
        ```

    Notes:
        - Spectral Slope: Ratio of energy in a spectra between 10-1000Hz over 1000-4000Hz.
        - Spectral Tilt: Linear slope of energy distribution between 100-5000Hz.
        - Using pitch-corrected LTAS to remove the effect of F0 and harmonics on the slope calculation:
        https://www.fon.hum.uva.nl/paul/papers/BoersmaKovacic2006.pdf
    """
    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        LTAS_rep = parselmouth.praat.call(
            snd, "To Ltas (pitch-corrected)...", floor, ceiling, 5000, 100, 0.0001, 0.02, 1.3
        )
        # Hyperparameters: Min Pitch (Hz), Max Pitch (Hz), Maximum Frequency (Hz), Bandwidth (Hz), Shortest Period (s),
        # Longest Period (s), Maximum period factor

        spc_slope = parselmouth.praat.call(LTAS_rep, "Get slope", 50, 1000, 1000, 4000, "dB")
        # Hyperparameters: f1min, f1max, f2min, f2max, averagingUnits

        spc_tilt_Report = parselmouth.praat.call(LTAS_rep, "Report spectral tilt", 100, 5000, "Linear", "Robust")
        # Hyperparameters: minimumFrequency, maximumFrequency, Frequency Scale (linear or logarithmic),
        # Fit method (least squares or robust)

        srt_st = spc_tilt_Report.index("Slope: ") + len("Slope: ")
        end_st = spc_tilt_Report.index("d", srt_st)
        spc_tilt = float(spc_tilt_Report[srt_st:end_st])

        # Return results
        return {"spc_slope": spc_slope, "spc_tilt": spc_tilt}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"spc_slope": float("nan"), "spc_Tilt": float("nan")}


def extract_cpp(
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

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_CPP(snd, 75, 500, 0.01)
        {'mean_cpp': 20.3}
        ```

    Notes:
        - Cepstral Peak Prominence: The height (i.e., “prominence”) of that peak relative to a regression line
        through the overall cepstrum.
        - Adapted from: https://osf.io/ctwgr and http://phonetics.linguistics.ucla.edu/facilities/acoustic/voiced_extract_auto.txt
    """
    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract pitch object for voiced checking
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling, voicing_threshold=0.3)

        pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

        textgrid = parselmouth.praat.call(pulses, "To TextGrid (vuv)", 0.02, 0.1)

        vuv_table = parselmouth.praat.call(textgrid, "Down to Table", "no", 6, "yes", "no")
        # Variables - include line number, Time decimals, include tier names, include empty intervals

        CPP_List = []

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
                    CPP_List.append(CPP_Value)

        # Calculate Final Features
        if CPP_List:
            CPP_array = np.array(CPP_List)
            CPP_mean = np.mean(CPP_array)
        else:
            CPP_mean = np.nan

        # Return Result
        return {"mean_cpp": CPP_mean}

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"mean_cpp": float("nan")}


def measure_formants(
    snd: Union[parselmouth.Sound, Path, Audio], floor: float, ceiling: float, frame_shift: float
) -> Dict[str, float]:
    """Extract Formant Frequency Features.

    Function to extract formant frequency features from a given sound object. This function is adapted from default
    Praat code to work with Parselmouth.

    Args:
        snd (Union[parselmouth.Sound, Path, Audio]): A Parselmouth Sound object or a file path or an Audio object.
        floor (float): Minimum expected pitch value, set using value found in `pitch_values` function.
        ceiling (float): Maximum expected pitch value, set using value found in `pitch_values` function.
        frame_shift (float): Time rate at which to extract a new pitch value, typically set to 5 ms.

    Returns:
        dict: A dictionary containing the following keys:

            - F1_mean (float): Mean F1 location.
            - F1_Std (float): Standard deviation of F1 location.
            - B1_mean (float): Mean F1 bandwidth.
            - B1_Std (float): Standard deviation of F1 bandwidth.
            - F2_mean (float): Mean F2 location.
            - F2_Std (float): Standard deviation of F2 location.
            - B2_mean (float): Mean F2 bandwidth.
            - B2_Std (float): Standard deviation of F2 bandwidth.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> measureFormants(snd, 75, 500, 0.01)
        {'F1_mean': 500.0, 'F1_Std': 50.0, 'B1_mean': 80.0, 'B1_Std': 10.0, 'F2_mean': 1500.0,
        'F2_Std': 100.0, 'B2_mean': 120.0, 'B2_Std': 20.0}
        ```

    Notes:
        - Formants are the resonances of the vocal tract, determined by tongue placement and vocal tract shape.
        - Mean F1 typically varies between 300 to 750 Hz, while mean F2 typically varies between 900 to 2300 Hz.
        - Formant bandwidth is measured by taking the width of the band forming 3 dB down from the formant peak.
        - Adapted from code at this [link](https://osf.io/6dwr3/).
    """
    try:
        if not isinstance(snd, parselmouth.Sound):
            snd = get_sound(snd)

        # Extract formants
        formants = parselmouth.praat.call(snd, "To Formant (burg)", frame_shift, 5, 5000, 0.025, 50)
        # Key Hyperparameters: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html

        # Extract pitch using CC method
        pitch = snd.to_pitch_cc(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

        F1_list, F2_list, B1_list, B2_list = [], [], [], []
        numPoints = parselmouth.praat.call(pulses, "Get number of points")

        for point in range(1, numPoints + 1):
            t = parselmouth.praat.call(pulses, "Get time from index", point)

            F1_value = parselmouth.praat.call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            if not np.isnan(F1_value):
                F1_list.append(F1_value)

            B1_value = parselmouth.praat.call(formants, "Get bandwidth at time", 1, t, "Hertz", "Linear")
            if not np.isnan(B1_value):
                B1_list.append(B1_value)

            F2_value = parselmouth.praat.call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            if not np.isnan(F2_value):
                F2_list.append(F2_value)

            B2_value = parselmouth.praat.call(formants, "Get bandwidth at time", 2, t, "Hertz", "Linear")
            if not np.isnan(B2_value):
                B2_list.append(B2_value)

        F1_mean, F1_Std = (np.mean(F1_list), np.std(F1_list)) if F1_list else (np.nan, np.nan)
        B1_mean, B1_Std = (np.mean(B1_list), np.std(B1_list)) if B1_list else (np.nan, np.nan)
        F2_mean, F2_Std = (np.mean(F2_list), np.std(F2_list)) if F2_list else (np.nan, np.nan)
        B2_mean, B2_Std = (np.mean(B2_list), np.std(B2_list)) if B2_list else (np.nan, np.nan)

        return {
            "F1_mean": F1_mean,
            "F1_Std": F1_Std,
            "B1_mean": B1_mean,
            "B1_Std": B1_Std,
            "F2_mean": F2_mean,
            "F2_Std": F2_Std,
            "B2_mean": B2_mean,
            "B2_Std": B2_Std,
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {
            "F1_mean": np.nan,
            "F1_Std": np.nan,
            "B1_mean": np.nan,
            "B1_Std": np.nan,
            "F2_mean": np.nan,
            "F2_Std": np.nan,
            "B2_mean": np.nan,
            "B2_Std": np.nan,
        }


def extract_Spectral_Moments(
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

            - spc_gravity (float): Mean spectral gravity.
            - spc_std_dev (float): Mean spectral standard deviation.
            - spc_skewness (float): Mean spectral skewness.
            - spc_kurtosis (float): Mean spectral kurtosis.

    Examples:
        ```python
        >>> snd = parselmouth.Sound("path_to_audio.wav")
        >>> extract_Spectral_Moments(snd, 75, 500, 0.025, 0.01)
        {'spc_gravity': 5000.0, 'spc_std_dev': 150.0, 'spc_skewness': -0.5, 'spc_kurtosis': 3.0}
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

        num_steps = parselmouth.praat.call(spectrogram, "Get number of frames")
        for i in range(1, num_steps + 1):
            t = parselmouth.praat.call(spectrogram, "Get time from frame number", i)
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

        Gravity_mean = np.mean(Gravity_list) if Gravity_list else np.nan
        STD_mean = np.mean(STD_list) if STD_list else np.nan
        Skew_mean = np.mean(Skew_list) if Skew_list else np.nan
        Kurt_mean = np.mean(Kurt_list) if Kurt_list else np.nan

        return {
            "spc_gravity": Gravity_mean,
            "spc_std_dev": STD_mean,
            "spc_skewness": Skew_mean,
            "spc_kurtosis": Kurt_mean,
        }

    except Exception as e:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            current_function_name = current_frame.f_code.co_name
            logger.error(f'Error in "{current_function_name}": \n' + str(e))
        return {"spc_gravity": np.nan, "spc_std_dev": np.nan, "spc_skewness": np.nan, "spc_kurtosis": np.nan}


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
    # Check if the input is a Path, in which case we load the audio from the file
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

    # Get the total duration of the sound
    duration = parselmouth.praat.call(snd, "Get total duration")

    # Return the duration in a dictionary
    return {"duration": duration}


### OK UNTIL HERE!!!!


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

    # Check if the input is a Path or Audio, and convert to Parselmouth Sound if necessary
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

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

    # Check if the input is a Path or Audio, and convert to Parselmouth Sound if necessary
    if not isinstance(snd, parselmouth.Sound):
        snd = get_sound(snd)

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


### Wrapper ###


def extract_features_from_audios(audios: list, 
                                        plugin: str = "cf",
                                        cache_dir: str = "./.pydra_cache") -> dict:
    """Extract features from a list of Audio objects and return a JSON-like dictionary.

    Args:
        audios (list): List of Audio objects to extract features from.
        plugin (str): Plugin to use for feature extraction. Defaults to "cf".
        cache_dir (str): Directory to use for caching by pydra. Defaults to "./.pydra_cache".

    Returns:
        dict: A JSON-like dictionary with extracted features structured under "praat_parselmouth".
    """
    # Mark tasks with Pydra
    extract_speech_rate_pt = pydra.mark.task(extract_speech_rate)
    extract_intensity_pt = pydra.mark.task(extract_intensity)
    extract_harmonicity_pt = pydra.mark.task(extract_harmonicity)
    measure_formants_pt = pydra.mark.task(measure_formants)
    extract_Spectral_Moments_pt = pydra.mark.task(extract_Spectral_Moments)
    extract_pitch_pt = pydra.mark.task(extract_pitch)
    extract_slope_tilt_pt = pydra.mark.task(extract_slope_tilt)
    extract_cpp_pt = pydra.mark.task(extract_cpp)
    extract_pitch_values_pt = pydra.mark.task(extract_pitch_values)
    extract_audio_duration_pt = pydra.mark.task(extract_audio_duration)

    def _extract_pitch_floor(pitch_values_out: dict) -> float:
        return pitch_values_out["pitch_floor"]
    _extract_pitch_floor_pt = pydra.mark.task(_extract_pitch_floor)

    def _extract_pitch_ceiling(pitch_values_out: dict) -> float:
        return pitch_values_out["pitch_ceiling"]
    _extract_pitch_ceiling_pt = pydra.mark.task(_extract_pitch_ceiling)

    # Create the workflow
    wf = pydra.Workflow(name="wf", input_spec=["x"], cache_dir=cache_dir)
    wf.split("x", x=audios)
    wf.add(extract_speech_rate_pt(name="extract_speech_rate_pt", snd=wf.lzin.x))
    wf.add(extract_pitch_values_pt(name="extract_pitch_values_pt", snd=wf.lzin.x))
    wf.add(
        _extract_pitch_floor_pt(name="_extract_pitch_floor_pt", pitch_values_out=wf.extract_pitch_values_pt.lzout.out)
    )
    wf.add(
        _extract_pitch_ceiling_pt(
            name="_extract_pitch_ceiling_pt", pitch_values_out=wf.extract_pitch_values_pt.lzout.out
        )
    )
    time_step = 0.005  # Feature Window Rate
    unit = "Hertz"
    wf.add(
        extract_pitch_pt(
            name="extract_pitch_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            ceiling=wf._extract_pitch_ceiling_pt.lzout.out,
            frame_shift=time_step,
            unit=unit
        )
    )
    wf.add(
        extract_intensity_pt(
            name="extract_intensity_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            frame_shift=time_step,
        )
    )
    wf.add(
        extract_harmonicity_pt(
            name="extract_harmonicity_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            frame_shift=time_step,
        )
    )
    wf.add(
        extract_slope_tilt_pt(
            name="extract_slope_tilt_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            ceiling=wf._extract_pitch_ceiling_pt.lzout.out,
        )
    )
    wf.add(
        extract_cpp_pt(
            name="extract_cpp_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            ceiling=wf._extract_pitch_ceiling_pt.lzout.out,
            frame_shift=time_step,
        )
    )
    wf.add(
        measure_formants_pt(
            name="measure_formants_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            ceiling=wf._extract_pitch_ceiling_pt.lzout.out,
            frame_shift=time_step,
        )
    )
    window_length = 0.025  # Length of feature extraction window for spectrogram
    wf.add(
        extract_Spectral_Moments_pt(
            name="extract_Spectral_Moments_pt",
            snd=wf.lzin.x,
            floor=wf._extract_pitch_floor_pt.lzout.out,
            ceiling=wf._extract_pitch_ceiling_pt.lzout.out,
            window_size=window_length,
            frame_shift=time_step,
        )
    )
    wf.add(extract_audio_duration_pt(name="extract_audio_duration_pt", snd=wf.lzin.x))

    # setting multiple workflow outputs
    wf.set_output(
        [
            ("speech_rate_out", wf.extract_speech_rate_pt.lzout.out),
            ("pitch_values_out", wf.extract_pitch_values_pt.lzout.out),
            ("pitch_out", wf.extract_pitch_pt.lzout.out),
            ("intensity_out", wf.extract_intensity_pt.lzout.out),
            ("harmonicity_out", wf.extract_harmonicity_pt.lzout.out),
            ("slope_tilt_out", wf.extract_slope_tilt_pt.lzout.out),
            ("cpp_out", wf.extract_cpp_pt.lzout.out),
            ("formants_out", wf.measure_formants_pt.lzout.out),
            ("spectral_moments_out", wf.extract_Spectral_Moments_pt.lzout.out),
            ("audio_duration", wf.extract_audio_duration_pt.lzout.out),
        ]
    )

    with pydra.Submitter(plugin=plugin) as sub:
        sub(wf)

    outputs = wf.result()

    print(outputs)

    extracted_data = []

    for output in outputs:
        feature_data = {
            # Audio duration
            "duration": output.output.audio_duration["duration"],
            # Timing and Pausing
            "speaking_rate": output.output.speech_rate_out["speaking_rate"],
            "articulation_rate": output.output.speech_rate_out["articulation_rate"],
            "phonation_ratio": output.output.speech_rate_out["phonation_ratio"],
            "pause_rate": output.output.speech_rate_out["pause_rate"],
            "mean_pause_duration": output.output.speech_rate_out["mean_pause_dur"],
            # Pitch and Intensity:
            f"mean_f0_{unit.lower()}": output.output.pitch_out[f"mean_f0_{unit.lower()}"],
            f"stdev_f0_{unit.lower()}": output.output.pitch_out[f"stdev_f0_{unit.lower()}"],
            "mean_db": output.output.intensity_out["mean_db"],
            "range_ratio_db": output.output.intensity_out["range_db_ratio"],
            # Quality Features:
            "hnr_db": output.output.harmonicity_out["HNR_db_mean"],
            "spectral_slope": output.output.slope_tilt_out["spc_slope"],
            "spectral_tilt": output.output.slope_tilt_out["spc_tilt"],
            "cepstral_peak_prominence": output.output.cpp_out["mean_cpp"],
            # Formant (F1, F2):
            "mean_f1_loc": output.output.formants_out["F1_mean"],
            "std_f1_loc": output.output.formants_out["F1_Std"],
            "mean_b1_loc": output.output.formants_out["B1_mean"],
            "std_b1_loc": output.output.formants_out["B1_Std"],
            "mean_f2_loc": output.output.formants_out["F2_mean"],
            "std_f2_loc": output.output.formants_out["F2_Std"],
            "mean_b2_loc": output.output.formants_out["B2_mean"],
            "std_b2_loc": output.output.formants_out["B2_Std"],
            # Spectral Moments:
            "spectral_gravity": output.output.spectral_moments_out["spc_gravity"],
            "spectral_std_dev": output.output.spectral_moments_out["spc_std_dev"],
            "spectral_skewness": output.output.spectral_moments_out["spc_skewness"],
            "spectral_kurtosis": output.output.spectral_moments_out["spc_kurtosis"],
        }

        extracted_data.append(feature_data)

    # Wrap extracted features in a dictionary under "praat_parselmouth"
    output_json = {"praat_parselmouth": extracted_data}

    return output_json
