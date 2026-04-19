"""Coqui TTS worker — runs inside the isolated subprocess venv.

This module is called by call_in_venv() and should NOT be imported
directly by senselab. It has access to coqui-tts and its own torch.
"""

import json
from pathlib import Path
from typing import Dict, List


def run_voice_cloning(
    source_paths: List[str],
    target_paths: List[str],
    model_id: str,
    device: str,
    output_dir: str,
) -> Dict[str, List[str]]:
    """Run voice cloning in the isolated coqui venv.

    Args:
        source_paths: Paths to source FLAC files.
        target_paths: Paths to target FLAC files.
        model_id: Coqui TTS model identifier.
        device: "cpu" or "cuda".
        output_dir: Directory for output FLAC files.

    Returns:
        Dict with "output_paths" list of cloned audio file paths.
    """
    import torchaudio
    from TTS.api import TTS

    tts = TTS(model_id).to(device=device)
    audio_config = tts.voice_converter.vc_config.audio

    output_sample_rate = (
        audio_config.output_sample_rate
        if hasattr(audio_config, "output_sample_rate")
        else getattr(audio_config, "sample_rate", None)
    )

    output_paths = []
    for i, (src_path, tgt_path) in enumerate(zip(source_paths, target_paths)):
        src_wav, src_sr = torchaudio.load(src_path)
        tgt_wav, tgt_sr = torchaudio.load(tgt_path)

        converted = tts.voice_conversion(
            source_wav=src_wav.squeeze(),
            target_wav=tgt_wav.squeeze(),
        )

        import torch

        if not isinstance(converted, torch.Tensor):
            converted = torch.tensor(converted)
        if converted.dim() == 1:
            converted = converted.unsqueeze(0)

        out_path = str(Path(output_dir) / f"cloned_{i}.flac")
        torchaudio.save(out_path, converted, output_sample_rate, format="flac")
        output_paths.append(out_path)

    return {"output_paths": output_paths}
