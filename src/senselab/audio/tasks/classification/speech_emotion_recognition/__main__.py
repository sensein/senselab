"""CLI helper to probe how `classify_emotions_from_speech` will route a given model.

Run via::

    python -m senselab.audio.tasks.classification.speech_emotion_recognition --probe <repo_id>

Reports:

- The encoder model_type (per ``config.model_type``).
- Architecture strings declared in the config.
- Whether the dispatcher would route through the custom head path, the subprocess
  venv, or the standard transformers pipeline.
- For the custom-head path: the head ``final_layer`` it would use.
- A warning if any assumption baked into the dispatcher could trip silently.

This does NOT load model weights — it only inspects config + checkpoint manifests.
"""

import argparse
import sys
from typing import Optional

from senselab.utils.data_structures import HFModel


def _probe(repo_id: str, revision: Optional[str] = None) -> int:
    """Print a routing report for ``repo_id``. Returns process exit code."""
    from senselab.audio.tasks.classification.speech_emotion_recognition.api import (
        _BASE_REGISTRY,
        _KNOWN_HEAD_LAYOUTS,
        _emotion_head_kind,
        _get_ser_type,
    )

    print(f"# Probe: {repo_id} (revision={revision or 'main'})")

    model: HFModel = HFModel(path_or_uri=repo_id, revision=revision or "main")

    # Config inspection
    config = None
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
        print(f"model_type:   {getattr(config, 'model_type', None)}")
        print(f"architectures: {getattr(config, 'architectures', None)}")
        print(f"auto_map:     {getattr(config, 'auto_map', None)}")
        print(f"num_labels:   {getattr(config, 'num_labels', None)}")
        print(f"problem_type: {getattr(config, 'problem_type', None)}")
        labels = list((getattr(config, "id2label", None) or {}).values())
        print(f"id2label:     {labels[:8]}{'…' if len(labels) > 8 else ''}")
    except Exception as e:  # pragma: no cover — diagnostic path
        print(f"AutoConfig.from_pretrained failed: {type(e).__name__}: {e}")
        print("(The dispatcher's raw-config-dict fallback handles this; routing still works.)")
        # ``config`` stays None below; getattr(None, ...) is safe and the advisories
        # that depend on config.model_type get skipped via short-circuit evaluation.

    # SER-type heuristic (handles AutoConfig failure via its own raw-dict fallback)
    ser_type = _get_ser_type(model)
    print(f"\nSERType heuristic: {ser_type.value}")

    # Head-kind detection
    head_match = _emotion_head_kind(model)
    if head_match is not None:
        head_model_type, head_entry = head_match
        print("\nDispatch: CUSTOM in-process emotion-head class")
        print(f"  encoder family: {head_model_type} (encoder attr: {_BASE_REGISTRY[head_model_type][2]})")
        print(f"  head:           {head_entry}")
        if repo_id in _KNOWN_HEAD_LAYOUTS:
            print(f"  source:         hardcoded entry in _KNOWN_HEAD_LAYOUTS[{repo_id!r}]")
        else:
            print("  source:         peeked checkpoint manifest (architectures + classifier keys)")
    else:
        if ser_type.value == "continuous":
            print("\nDispatch: SUBPROCESS VENV (continuous SER + huggingface_hub<1.0 pin)")
        else:
            print("\nDispatch: STANDARD transformers pipeline()")
            print("  Phase-2 head-load check will warn if the checkpoint's head weights are missing or mis-shaped.")
            print("  Set SENSELAB_STRICT_HEAD_LOAD=1 to promote the warning to a hard error.")

    # Emit advisories
    print("\nAdvisories:")
    if ser_type.value == "not_recognized":
        print("  - Labels don't match the English emotion-keyword vocabulary; the dispatcher will reject this")
        print("    model unless its HF tags include 'speech-emotion-recognition' or 'emotion-recognition'.")
    if head_match is None and getattr(config, "model_type", None) in _BASE_REGISTRY:
        print("  - Encoder family is supported but the head layout was not detected. If this checkpoint")
        print("    has a 2-layer dense+linear head with an unrecognized final-layer name, add it to")
        print("    _KNOWN_HEAD_LAYOUTS or _FINAL_LAYER_NAMES in the api module.")
    cfg_model_type = getattr(config, "model_type", None)
    if head_match is None and cfg_model_type and cfg_model_type not in _BASE_REGISTRY:
        print(f"  - Encoder model_type={cfg_model_type!r} is not in _BASE_REGISTRY. Custom-head loading")
        print("    is unavailable for this family; falls back to the standard pipeline.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m senselab.audio.tasks.classification.speech_emotion_recognition",
        description="Probe a speech emotion recognition checkpoint without loading weights.",
    )
    parser.add_argument(
        "--probe",
        metavar="REPO_ID",
        required=True,
        help="HuggingFace repo id to probe (no inference run).",
    )
    parser.add_argument("--revision", default=None, help="Optional revision (default: main).")
    args = parser.parse_args(argv)
    return _probe(args.probe, args.revision)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
