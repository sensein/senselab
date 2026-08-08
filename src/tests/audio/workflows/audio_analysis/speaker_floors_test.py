"""A cosine calibration band is per-embedder, not per-pass.

A band is a property of the embedding space: ecapa's same/different separation is not resnet's. One
pass-level pair calibrated *every* embedder's distances with whichever model the clustering loop kept —
silently, because that loop kept only the first.
"""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.speaker import harvest_speaker_votes


def _windows(n: int = 8) -> list[WindowEmbedding]:
    """Alternating vectors at a *moderate* angle — cosine distance ~0.13.

    Deliberately not orthogonal. A distance of 1.0 sits above the diff-floor of every plausible band,
    so both bands saturate at the same value and the test cannot tell them apart. 0.13 falls *inside*
    one band and *below* the other, which is what makes the calibration visible.
    """
    out = []
    near = np.array([0.866, 0.5], dtype=np.float64)
    for i in range(n):
        vec = np.array([1.0, 0.0], dtype=np.float64) if i % 2 == 0 else near
        out.append(WindowEmbedding(start_s=i * 0.25, end_s=i * 0.25 + 0.25, vector=vec))
    return out


def _pass_summary() -> dict:
    return {
        "duration_s": 2.0,
        "diarization": {
            "by_model": {
                "pyannote": {
                    "status": "ok",
                    "result": [[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]],
                }
            }
        },
    }


def test_each_embedder_is_calibrated_with_its_own_band() -> None:
    """Two embedders, two bands, and the votes differ accordingly.

    With one band applied to both, the looser embedder's distances were read against the stricter
    model's separation — a calibration borrowed from a model that did not produce it.
    """
    embeddings = {"ecapa": _windows(), "resnet": _windows()}
    buckets = harvest_speaker_votes(
        pass_summary=_pass_summary(),
        grid=BucketGrid(win_length=0.25, hop_length=0.25),
        per_window_embeddings=embeddings,
        speaker_floors={"ecapa": (0.10, 0.20), "resnet": (0.80, 0.95)},
    )
    votes = {k: v for b in buckets for k, v in b["votes"].items() if "::" in k}
    ecapa = [v for k, v in votes.items() if k.endswith("::ecapa")]
    resnet = [v for k, v in votes.items() if k.endswith("::resnet")]
    assert ecapa and resnet, "both embedders should produce validation readings"
    # ``calibrated_same_doubt`` was ``same_label_uncertainty``. The per-embedder band still produces
    # it and L1 still records it; it is simply no longer scored by the fold, since the axis measures
    # attribution rather than change.
    # Identical vectors, different bands → different calibrated uncertainties.
    e_unc = [v.get("calibrated_same_doubt") for v in ecapa if v.get("calibrated_same_doubt") is not None]
    r_unc = [v.get("calibrated_same_doubt") for v in resnet if v.get("calibrated_same_doubt") is not None]
    assert e_unc and r_unc
    # 0.13 is inside ecapa's (0.10, 0.20) ramp and below resnet's same-floor of 0.80, so ecapa reads
    # partial doubt where resnet reads confident agreement. One band for both collapsed that.
    assert e_unc != r_unc, "one band for both embedders is what this replaces"


def test_an_embedder_with_no_measured_band_falls_back_to_the_default() -> None:
    """Not measured is not zero, and not somebody else's band either."""
    embeddings = {"ecapa": _windows(), "resnet": _windows()}
    buckets = harvest_speaker_votes(
        pass_summary=_pass_summary(),
        grid=BucketGrid(win_length=0.25, hop_length=0.25),
        per_window_embeddings=embeddings,
        same_speaker_floor=0.30,
        diff_speaker_floor=0.70,
        speaker_floors={"ecapa": (0.10, 0.20)},
    )
    keys = {k for b in buckets for k in b["votes"] if "::" in k}
    assert any(k.endswith("::resnet") for k in keys), "the unmeasured embedder still votes, on defaults"


def test_no_bands_at_all_still_votes_on_the_defaults() -> None:
    """The CLI defaults are the floor of last resort, not an absence of calibration."""
    buckets = harvest_speaker_votes(
        pass_summary=_pass_summary(),
        grid=BucketGrid(win_length=0.25, hop_length=0.25),
        per_window_embeddings={"ecapa": _windows()},
        same_speaker_floor=0.30,
        diff_speaker_floor=0.70,
    )
    assert any("::" in k for b in buckets for k in b["votes"])
