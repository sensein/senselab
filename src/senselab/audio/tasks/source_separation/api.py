"""Public API for the senselab source separation task.

Exposes class-space resolution for unasdiff's sound prior (:func:`resolve_source_classes`) and
the separation entry point itself, :func:`separate_audios`, which currently has one backend
(unasdiff) reachable only by calling this function directly -- see
``senselab.audio.tasks.source_separation.unasdiff``'s module docstring for the licensing reason
it is not wired into any default model list or into ``audio_analysis``.
"""

from __future__ import annotations

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.source_separation.unasdiff import (
    _DIFFUSION_STEPS,
    load_fsd_class_map_document,
    separate_with_unasdiff,
)
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel

# unasdiff is the only backend this task has, so this is a containment check (reject anything
# that isn't naming this backend) rather than a dispatch table like enhancement/api.py's -- see
# senselab.utils.data_structures.model.model_for_task's "separation" branch, which duplicates
# this literal by hand for the same layering reason its own Note documents for "enhancement".
_UNASDIFF_MODEL_PREFIX = "sensein/unasdiff"

_MODE_SPEECH_SOUND = "speech_sound"
_MODE_SOUND_SOUND = "sound_sound"
_MODE_SPEECH_SPEECH = "speech_speech"
_VALID_MODES = (_MODE_SPEECH_SOUND, _MODE_SOUND_SOUND, _MODE_SPEECH_SPEECH)


def separate_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    n_sources: int = 2,
    mode: str = _MODE_SPEECH_SOUND,
    source_classes: Optional[List[str]] = None,
    device: Optional[DeviceType] = None,
    seed: int = 17,
    diffusion_steps: int = _DIFFUSION_STEPS,
) -> List[List[Audio]]:
    """Separate each audio into ``n_sources`` sources with unasdiff.

    Three modes, matching unasdiff's two independently-trained diffusion priors:

    - ``"speech_sound"``: slot 0 is the speech prior (unconditional); the remaining
      ``n_sources - 1`` slots are the sound prior, conditioned on ``source_classes``.
    - ``"sound_sound"``: every slot is the sound prior, conditioned on ``source_classes``
      (one name per slot).
    - ``"speech_speech"``: every slot is the speech prior. No ``source_classes`` needed --
      the speech prior's label space has exactly one member.

      ``speech_speech`` ships with a caveat from unasdiff's own README:

          "The source-model-based separation approach is not well suited for
          same-class source separation (e.g. speech separation), because it lacks
          speaker-conditioning. In future work, we will attempt to address such
          issue."

      It is exposed because the alternative is a user rediscovering the limitation
      by measurement, but nothing downstream should treat its output as a reliable
      decomposition.

    Args:
        audios: Inputs. Resampled and downmixed to 16 kHz mono if needed.
        model: If given, must be an ``HFModel`` whose ``path_or_uri`` names the unasdiff
            checkpoint mirror (prefix ``"sensein/unasdiff"``). Source separation has exactly
            one backend today, so this is a containment check, not a dispatch: it exists so a
            caller cannot accidentally reach this backend through a model id meant for another
            task. ``None`` (the default) uses the pinned mirror unconditionally.
        n_sources: Number of sources to separate into.
        mode: One of ``"speech_sound"``, ``"sound_sound"``, ``"speech_speech"``.
        source_classes: FSD sound-prior class names, one per sound slot (``n_sources - 1`` for
            ``"speech_sound"``, ``n_sources`` for ``"sound_sound"``, unused for
            ``"speech_speech"``). Resolved to conditioning indices via
            :func:`resolve_source_classes`.
        device: Device the worker runs on -- CUDA or CPU. ``None`` leaves the choice to the
            worker. Forwarded unchanged to
            :func:`senselab.audio.tasks.source_separation.unasdiff.separate_with_unasdiff`.
        seed: RNG seed, recorded in the worker's log line.
        diffusion_steps: Number of reverse-diffusion steps the sampler runs per window --
            forwarded unchanged to :func:`senselab.audio.tasks.source_separation.unasdiff.separate_with_unasdiff`.
            Defaults to ``200``, upstream's own quality setting and the only value with any
            published basis; lower values trade quality for speed roughly proportionally but are
            entirely unmeasured in this repository, so there is no recommended lower setting --
            see that function's docstring for the full derivation.

    Returns:
        One list of ``n_sources`` ``Audio`` objects per input, in order.

    Raises:
        ValueError: If ``model`` is given and does not name the unasdiff mirror; if ``mode`` is
            not one of the three supported modes; if a mode using the sound prior is missing
            ``source_classes``; if ``source_classes`` does not have exactly the number of
            entries that mode's sound slots require; or if ``diffusion_steps`` is not positive.
    """
    if model is not None and not (
        isinstance(model, HFModel) and str(model.path_or_uri).startswith(_UNASDIFF_MODEL_PREFIX)
    ):
        raise ValueError(
            "source separation currently has one backend (unasdiff); model must be None or an "
            f"HFModel whose path_or_uri starts with {_UNASDIFF_MODEL_PREFIX!r}, got {model!r}"
        )
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes are: {', '.join(_VALID_MODES)}")

    if mode == _MODE_SPEECH_SPEECH:
        # Both slots are the speech prior; its label space has exactly one member (0), so
        # there is nothing for a caller to name.
        source_class_indices = [0] * n_sources
    else:
        n_sound_slots = n_sources - 1 if mode == _MODE_SPEECH_SOUND else n_sources
        if not source_classes:
            raise ValueError(
                f"mode={mode!r} conditions on unasdiff's sound prior, which has no defensible "
                "default class (index 0 is 'Hi-hat'); pass source_classes."
            )
        if len(source_classes) != n_sound_slots:
            raise ValueError(
                f"source_classes must have exactly {n_sound_slots} entries for n_sources="
                f"{n_sources} in mode={mode!r}; got {len(source_classes)}"
            )
        sound_indices = resolve_source_classes(source_classes)
        source_class_indices = [0, *sound_indices] if mode == _MODE_SPEECH_SOUND else sound_indices

    return separate_with_unasdiff(
        audios,
        n_sources,
        source_class_indices,
        mode=mode,
        device=device,
        seed=seed,
        diffusion_steps=diffusion_steps,
    )


def resolve_source_classes(names: list[str]) -> list[int]:
    """Resolve FSD sound-prior class names to their conditioning indices.

    Args:
        names: Class names as they appear in the FSD class map (e.g.
            ``["Applause", "Cello"]``).

    Returns:
        The conditioning index for each name, in the same order.

    Raises:
        ValueError: If any name is not in the class map. The sound prior's
            embedding has 50 slots but only 41 were trained (see
            ``data/fsd41_classes.json``'s derivation); silently mapping an
            unknown name to a fallback index would condition the prior on
            whatever class that fallback happens to name, while the caller's
            requested label -- and the output -- would go on reporting
            something else. The message enumerates the valid names so the
            caller can fix the typo without a second round trip.
    """
    classes = load_fsd_class_map_document()["classes"]
    unknown = [name for name in names if name not in classes]
    if unknown:
        valid = ", ".join(sorted(classes))
        raise ValueError(f"Unknown source class name(s): {', '.join(unknown)!r}. Valid classes are: {valid}")
    return [classes[name] for name in names]
