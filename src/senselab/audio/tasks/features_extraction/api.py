"""This module provides functions for extracting features from audio files."""

from typing import Any, Dict, List

import pydra

from senselab.audio.data_structures import Audio

from .opensmile import extract_opensmile_features_from_audios
from .praat_parselmouth import extract_praat_parselmouth_features_from_audios
from .torchaudio import extract_torchaudio_features_from_audios
from .torchaudio_squim import extract_objective_quality_features_from_audios


def extract_features_from_audios(audios: List[Audio], plugin: str = "cf") -> List[Dict[str, Any]]:
    """Extract features from a list of audio objects.

    Args:
        audios (List[Audio]): The list of audio objects to extract features from.
        plugin (str): The feature extraction plugin (default is "cf").

    Returns:
        List[Dict[str, Any]]: The list of feature dictionaries for each audio.
    """
    # opensmile
    extract_opensmile_features_from_audios_pt = pydra.mark.task(extract_opensmile_features_from_audios)
    # praat_parselmouth
    extract_praat_parselmouth_features_from_audios_pt = pydra.mark.task(extract_praat_parselmouth_features_from_audios)
    # torchaudio
    extract_torchaudio_features_from_audios_pt = pydra.mark.task(extract_torchaudio_features_from_audios)
    # torchaudio_squim
    extract_objective_quality_features_from_audios_pt = pydra.mark.task(extract_objective_quality_features_from_audios)

    formatted_audios = [[audio] for audio in audios]

    wf = pydra.Workflow(name="wf", input_spec=["x"])
    wf.split("x", x=formatted_audios)
    wf.add(
        extract_opensmile_features_from_audios_pt(name="extract_opensmile_features_from_audios_pt", audios=wf.lzin.x)
    )
    wf.add(
        extract_praat_parselmouth_features_from_audios_pt(
            name="extract_praat_parselmouth_features_from_audios_pt", audios=wf.lzin.x
        )
    )
    wf.add(
        extract_torchaudio_features_from_audios_pt(name="extract_torchaudio_features_from_audios_pt", audios=wf.lzin.x)
    )
    wf.add(
        extract_objective_quality_features_from_audios_pt(
            name="extract_objective_quality_features_from_audios_pt", audio_list=wf.lzin.x
        )
    )

    # setting multiple workflow outputs
    wf.set_output(
        [
            ("opensmile_out", wf.extract_opensmile_features_from_audios_pt.lzout.out),
            ("praat_parselmouth_out", wf.extract_praat_parselmouth_features_from_audios_pt.lzout.out),
            ("torchaudio_out", wf.extract_torchaudio_features_from_audios_pt.lzout.out),
            ("torchaudio_squim_out", wf.extract_objective_quality_features_from_audios_pt.lzout.out),
        ]
    )

    with pydra.Submitter(plugin=plugin) as sub:
        sub(wf)

    outputs = wf.result()

    formatted_output = []
    for output in outputs:
        formatted_output_item = {
            "opensmile": output.output.opensmile_out[0],
            "praat_parselmouth": output.output.praat_parselmouth_out["praat_parselmouth"][0],
            "torchaudio": output.output.torchaudio_out[0]["torchaudio"],
            "torchaudio_squim": {
                "stoi": output.output.torchaudio_squim_out["stoi"][0],
                "pesq": output.output.torchaudio_squim_out["pesq"][0],
                "si_sdr": output.output.torchaudio_squim_out["si_sdr"][0],
            },
        }

        formatted_output.append(formatted_output_item)

    return formatted_output
