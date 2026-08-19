"""SpeechScore: ClearerVoice-Studio's speech-quality metrics, 18 families over one audio.

Fourteen need a clean reference; four do not, which is what makes them usable on a real recording
where no reference exists. :data:`SPEECHSCORE_METRICS` is the classification, and it is senselab's own:
upstream's ``ScoreBasis.intrusive`` attribute is never read by its own code and disagrees with its own
documentation for several metrics.

SpeechScore has no pip distribution — the one on PyPI is an unrelated package by another author — so it
arrives as a sparse clone of the upstream repository at a pinned commit, which is also what pins the
metric weights, since NISQA's, DNSMOS's and DISTILL_MOS's are committed alongside the code. It gets its
own venv, disjoint from ``clearvoice``'s.

Design, the pin, and the two upstream traps this module works around:
``specs/20260819-clearvoice-integration/design.md`` §9.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from senselab.audio.data_structures import Audio
from senselab.utils.clearvoice import (
    SPEECHSCORE_COMMIT,
    SPEECHSCORE_PYTHON,
    SPEECHSCORE_REPO_URL,
    SPEECHSCORE_REQUIREMENTS,
    SPEECHSCORE_VENV,
)
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    venv_python,
)

# Terms of the default ceiling: seconds per (audio-second x metric). Three of the eighteen are neural
# forward passes and BSSEval/MCD are the slowest of the rest; unmeasured on this branch. design.md §9.
_SECONDS_PER_AUDIO_SECOND_PER_METRIC = 2.0
_TIMEOUT_FLOOR_S = 900.0


@dataclass(frozen=True)
class SpeechScoreMetric:
    """One SpeechScore metric.

    Attributes:
        name: The key SpeechScore's own factory dispatches on, and the key in the returned dict.
        needs_reference: Whether the metric compares against a clean reference signal.
        fields: Sub-keys the metric returns, for the metrics that return a mapping rather than a
            scalar. Empty means a scalar.
        description: What the metric measures.
    """

    name: str
    needs_reference: bool
    fields: Tuple[str, ...]
    description: str


SPEECHSCORE_METRICS: Dict[str, SpeechScoreMetric] = {
    metric.name: metric
    for metric in (
        # No reference required.
        SpeechScoreMetric("DNSMOS", False, ("SIG", "BAK", "OVRL", "P808_MOS"), "DNS-Challenge MOS predictor"),
        SpeechScoreMetric(
            "NISQA",
            False,
            ("mos_pred", "noi_pred", "col_pred", "dis_pred", "loud_pred"),
            "NISQA MOS plus noisiness, colouration, discontinuity and loudness",
        ),
        SpeechScoreMetric("DISTILL_MOS", False, (), "compact distilled MOS predictor"),
        SpeechScoreMetric("SRMR", False, (), "speech-to-reverberation modulation energy ratio"),
        # Reference required.
        SpeechScoreMetric("PESQ", True, (), "wideband perceptual evaluation of speech quality"),
        SpeechScoreMetric("NB_PESQ", True, (), "narrowband PESQ"),
        SpeechScoreMetric("STOI", True, (), "short-time objective intelligibility"),
        SpeechScoreMetric("SISDR", True, (), "scale-invariant signal-to-distortion ratio"),
        SpeechScoreMetric("SNR", True, (), "signal-to-noise ratio"),
        SpeechScoreMetric("SSNR", True, (), "segmental signal-to-noise ratio"),
        SpeechScoreMetric("FWSEGSNR", True, (), "frequency-weighted segmental SNR"),
        SpeechScoreMetric("LSD", True, (), "log-spectral distance"),
        SpeechScoreMetric("LLR", True, (), "log-likelihood ratio"),
        SpeechScoreMetric("CSIG", True, (), "composite signal-distortion rating"),
        SpeechScoreMetric("CBAK", True, (), "composite background-intrusiveness rating"),
        SpeechScoreMetric("COVL", True, (), "composite overall-quality rating"),
        SpeechScoreMetric("MCD", True, (), "mel-cepstral distortion"),
        SpeechScoreMetric("BSSEval", True, ("ISR", "SAR", "SDR"), "BSS_Eval isolation, artefacts and distortion"),
    )
}

NO_REFERENCE_METRICS: Tuple[str, ...] = tuple(
    name for name, metric in SPEECHSCORE_METRICS.items() if not metric.needs_reference
)
REFERENCE_METRICS: Tuple[str, ...] = tuple(name for name, metric in SPEECHSCORE_METRICS.items() if metric.needs_reference)


_WORKER_SCRIPT = r"""
import json
import os
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    repo_dir = Path(args["repo_dir"])

    import fcntl
    import shutil
    import subprocess as sp
    import tempfile as _tempfile

    # A blobless, sparse clone at a pinned commit: the studio's other directories carry checkpoints
    # this module never reads. Cloned under an exclusive lock into a sibling temp dir and moved into
    # place, so an interrupted clone cannot leave a directory that looks complete.
    marker = repo_dir / "speechscore" / "speechscore.py"
    if not marker.is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(str(repo_dir) + ".lock", "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not marker.is_file():
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                tmp_clone = Path(_tempfile.mkdtemp(prefix=".speechscore-clone-", dir=str(repo_dir.parent)))
                try:
                    sp.run(["git", "init", "-q", str(tmp_clone)], check=True)
                    sp.run(["git", "-C", str(tmp_clone), "remote", "add", "origin", args["repo_url"]], check=True)
                    sp.run(["git", "-C", str(tmp_clone), "config", "core.sparseCheckout", "true"], check=True)
                    sp.run(
                        ["git", "-C", str(tmp_clone), "sparse-checkout", "set", "--no-cone", "/speechscore/"],
                        check=True,
                    )
                    sp.run(
                        ["git", "-C", str(tmp_clone), "fetch", "-q", "--depth", "1", "origin", args["commit"]],
                        check=True,
                    )
                    sp.run(["git", "-C", str(tmp_clone), "checkout", "-q", "FETCH_HEAD"], check=True)
                except Exception:
                    shutil.rmtree(tmp_clone, ignore_errors=True)
                    raise
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                os.replace(tmp_clone, repo_dir)

    # Two constraints, both load-bearing. The metric weights are addressed relative to the working
    # directory ("scores/nisqa/weights/nisqa.tar"), so the cwd must be the speechscore directory; and
    # that directory -- not its parent -- must be on sys.path, because speechscore/__init__.py imports
    # modules named "absolute" and "relative" that do not exist, so importing it as a package fails.
    package_dir = repo_dir / "speechscore"
    os.chdir(package_dir)
    sys.path.insert(0, str(package_dir))

    from speechscore import SpeechScore

    scorer = SpeechScore(args["metrics"])

    results = []
    for test_path, reference_path in zip(args["test_paths"], args["reference_paths"]):
        # window=None always: upstream's windowed branch in basis.py references an undefined name and
        # raises NameError, so windowing is not a capability this can offer.
        results.append(
            scorer(
                test_path=test_path,
                reference_path=reference_path,
                window=None,
                score_rate=args["score_rate"],
                return_mean=False,
            )
        )

    print(json.dumps({"results": results}, default=float))
except Exception as exc:
    import traceback

    print(
        json.dumps(
            {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            }
        )
    )
    sys.exit(1)
"""


def default_speechscore_timeout_s(total_audio_s: float, n_metrics: int) -> float:
    """Return the default worker ceiling for scoring ``total_audio_s`` with ``n_metrics`` metrics.

    Args:
        total_audio_s: Total duration to be scored, summed over every input.
        n_metrics: How many metrics were requested.

    Returns:
        Seconds, never below :data:`_TIMEOUT_FLOOR_S`.
    """
    return max(_TIMEOUT_FLOOR_S, _SECONDS_PER_AUDIO_SECOND_PER_METRIC * total_audio_s * max(n_metrics, 1))


def resolve_speechscore_metrics(metrics: Optional[Sequence[str]], has_references: bool) -> List[str]:
    """Validate a metric selection against what a reference-free call can compute.

    Args:
        metrics: Metric names, case-insensitive. ``None`` selects every metric the call can compute:
            all eighteen with references, the four no-reference ones without.
        has_references: Whether reference audios were supplied.

    Returns:
        Metric names in :data:`SPEECHSCORE_METRICS` order.

    Raises:
        ValueError: If a name is unknown, or if a reference-requiring metric was asked for with no
            reference. Upstream computes such a metric against a zero-padded copy of the test signal
            and returns a plausible number, so this must refuse rather than pass the request through.
    """
    if metrics is None:
        selected = list(SPEECHSCORE_METRICS) if has_references else list(NO_REFERENCE_METRICS)
        return selected

    canonical = {name.upper(): name for name in SPEECHSCORE_METRICS}
    resolved: List[str] = []
    unknown: List[str] = []
    for requested in metrics:
        name = canonical.get(str(requested).upper())
        if name is None:
            unknown.append(str(requested))
        else:
            resolved.append(name)
    if unknown:
        raise ValueError(
            f"Unknown SpeechScore metric(s): {', '.join(repr(name) for name in unknown)}. Known: "
            f"{', '.join(SPEECHSCORE_METRICS)}."
        )

    if not has_references:
        needing = [name for name in resolved if SPEECHSCORE_METRICS[name].needs_reference]
        if needing:
            raise ValueError(
                f"{', '.join(needing)} compare against a clean reference, and none was given. Upstream "
                "would score them against a copy of the test signal and return a plausible number. "
                f"Pass reference_audios, or choose from {', '.join(NO_REFERENCE_METRICS)}."
            )
    # Preserve table order, so a result dict's keys do not depend on the caller's argument order.
    return [name for name in SPEECHSCORE_METRICS if name in set(resolved)]


def extract_speechscore_metrics_from_audios(
    audios: List[Audio],
    reference_audios: Optional[List[Audio]] = None,
    metrics: Optional[Sequence[str]] = None,
    score_rate: int = 16000,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Score each audio with SpeechScore, optionally against a reference.

    Args:
        audios: The audios to score.
        reference_audios: Clean references, one per audio in the same order. ``None`` restricts the
            selection to the four metrics that need none.
        metrics: Metric names, case-insensitive; ``None`` takes everything the call can compute.
        score_rate: Rate passed to SpeechScore for metrics that do not fix their own. Each metric that
            declares a rate resamples to it regardless.
        timeout_s: Ceiling on the worker, in seconds. ``None`` derives one from the duration and the
            number of metrics.

    Returns:
        One dict per audio, keyed by metric name in :data:`SPEECHSCORE_METRICS` order. A metric with
        sub-fields maps to a nested dict.

    Raises:
        ValueError: If ``reference_audios`` is given but does not match ``audios`` in length, if a
            metric name is unknown or needs a reference that was not given, or if ``timeout_s`` is not
            positive.
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    if not audios:
        return []
    if reference_audios is not None and len(reference_audios) != len(audios):
        raise ValueError(
            f"reference_audios must have one entry per audio: got {len(reference_audios)} for {len(audios)} audios"
        )
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s}")

    selected = resolve_speechscore_metrics(metrics, has_references=reference_audios is not None)
    total_audio_s = sum(audio.waveform.shape[-1] / audio.sampling_rate for audio in audios)
    effective_timeout_s = (
        default_speechscore_timeout_s(total_audio_s, len(selected)) if timeout_s is None else timeout_s
    )

    venv_dir = ensure_venv(SPEECHSCORE_VENV, SPEECHSCORE_REQUIREMENTS, python_version=SPEECHSCORE_PYTHON)
    python = venv_python(venv_dir)
    # Cached beside the venv, so the pinned clone happens once per host rather than once per call.
    repo_dir = Path(venv_dir) / "clearervoice-studio-src"

    logger.info(
        "SpeechScore: %d metric(s) over %d audio(s), %.10gs, commit %s, timeout=%.10gs",
        len(selected),
        len(audios),
        total_audio_s,
        SPEECHSCORE_COMMIT[:12],
        effective_timeout_s,
    )

    with tempfile.TemporaryDirectory(prefix="senselab-speechscore-") as tmpdir:
        tmp = Path(tmpdir)
        test_paths, reference_paths = [], []
        for index, audio in enumerate(audios):
            test_path = str(tmp / f"test_{index}.wav")
            audio.save_to_file(test_path)
            test_paths.append(test_path)
            if reference_audios is None:
                reference_paths.append(None)
            else:
                reference_path = str(tmp / f"ref_{index}.wav")
                reference_audios[index].save_to_file(reference_path)
                reference_paths.append(reference_path)

        try:
            result = subprocess.run(
                [python, "-c", _WORKER_SCRIPT],
                input=json.dumps(
                    {
                        "repo_dir": str(repo_dir),
                        "repo_url": SPEECHSCORE_REPO_URL,
                        "commit": SPEECHSCORE_COMMIT,
                        "metrics": selected,
                        "test_paths": test_paths,
                        "reference_paths": reference_paths,
                        "score_rate": score_rate,
                    }
                ),
                capture_output=True,
                text=True,
                timeout=effective_timeout_s,
                env=_clean_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"SpeechScore worker exceeded its {effective_timeout_s:.10g}s ceiling scoring "
                f"{total_audio_s:.10g}s of audio with {len(selected)} metric(s). No score is returned. "
                "Pass timeout_s to raise the ceiling, or request fewer metrics — NISQA, DNSMOS and "
                "DISTILL_MOS are neural and dominate the cost."
            ) from exc
        output = parse_subprocess_result(result, venv_label="SpeechScore")

    return [{name: scores[name] for name in selected if name in scores} for scores in output["results"]]
