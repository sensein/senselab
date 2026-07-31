#!/usr/bin/env python3
"""Fit a scene-quality calibration profile from synthetic mixtures (US5, T037/T038).

Synthesizes controlled degradations of a clean reference clip — additive
white/pink noise across an SNR sweep and synthetic exponential-decay RIRs across
an RT60 sweep — runs the workflow's quality estimators on each variant, fits the
dB→[0,1] normalization anchors, and persists a versioned
``CalibrationProfile`` (data-model §5) plus a reported-vs-true validation plot
and table (FR-022 / SC-007).

Fitting scope (documented): the dB anchors (SNR always; C50 when Brouhaha is
available — the DSP path has no reverb estimator, in which case the C50 block
keeps its documented defaults and the provenance says so). The per-axis
``temperature`` entries are CLI passthroughs (default 1.0): a proper temperature
fit needs labeled correctness (e.g. the adaptive loop's ground-truth harness),
not a synthetic sweep.

Usage (full senselab environment):
    uv run python scripts/calibrate_scene_quality.py \
        --audio tutorial_audio_files/audio_48khz_mono_16bits.wav \
        --out artifacts/scene_quality_calibration.json
Then pass it to the pipeline:
    uv run python scripts/analyze_audio.py clip.wav \
        --calibration-profile artifacts/scene_quality_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI (constitution VIII: every input/output/sweep is a parameter with a default)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, default=Path("tutorial_audio_files/audio_48khz_mono_16bits.wav"))
    parser.add_argument("--snr-sweep-db", type=float, nargs="+", default=[30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0])
    parser.add_argument("--rt60-sweep-s", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2])
    parser.add_argument("--noise", choices=("white", "pink"), default="white")
    parser.add_argument("--clean-anchor-db", type=float, default=25.0, help="True SNR mapped to degradation 0")
    parser.add_argument("--floor-anchor-db", type=float, default=5.0, help="True SNR mapped to degradation 1")
    parser.add_argument("--temperature-speech-presence", type=float, default=1.0)
    parser.add_argument("--temperature-asr", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/scene_quality_calibration.json"))
    parser.add_argument("--plot", type=Path, default=Path("artifacts/scene_quality_calibration_validation.png"))
    parser.add_argument("--table", type=Path, default=Path("artifacts/scene_quality_calibration_validation.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-brouhaha", action="store_true", help="Skip the Brouhaha C50 sweep (DSP-only fit)")
    return parser.parse_args(argv)


def _synth_noise(shape: int, kind: str, rng: Any) -> Any:  # noqa: ANN401 — np.ndarray
    import numpy as np

    white = rng.normal(0.0, 1.0, shape)
    if kind == "white":
        return white
    # Pink: 1/f shaping in the frequency domain.
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(shape)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    pink = np.fft.irfft(spectrum / np.sqrt(freqs), n=shape)
    return pink / max(1e-12, float(np.std(pink)))


def _mix_at_snr(speech: Any, noise: Any, snr_db: float) -> Any:  # noqa: ANN401
    import numpy as np

    speech_power = float(np.mean(speech**2))
    noise_power = float(np.mean(noise**2))
    scale = np.sqrt(speech_power / (10 ** (snr_db / 10)) / max(1e-12, noise_power))
    return np.clip(speech + scale * noise, -1.0, 1.0).astype("float32")


def _exponential_rir(rt60_s: float, sr: int, rng: Any, length_s: float = 1.5) -> Any:  # noqa: ANN401
    import numpy as np

    n = max(1, int(length_s * sr))
    t = np.arange(n) / sr
    decay = np.exp(-6.9077 * t / max(1e-3, rt60_s))  # -60 dB at rt60
    h = decay * rng.normal(0.0, 1.0, n)
    h[0] = 1.0
    return h / max(1e-12, float(np.max(np.abs(h))))


def _true_c50_db(h: Any, sr: int) -> float:  # noqa: ANN401
    import numpy as np

    k = int(0.05 * sr)
    early = float(np.sum(h[:k] ** 2))
    late = max(1e-12, float(np.sum(h[k:] ** 2)))
    return 10.0 * float(np.log10(max(1e-12, early) / late))


def _estimate(audio_np: Any, sr: int, *, brouhaha: bool) -> dict[str, float | None]:  # noqa: ANN401
    """Median raw estimator outputs (dB) over the clip via the workflow's quality harvest."""
    import numpy as np
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.quality import harvest_quality_measurements

    audio = Audio(waveform=torch.from_numpy(audio_np).unsqueeze(0), sampling_rate=sr)
    brouhaha_frames = None
    if brouhaha:
        from senselab.audio.tasks.scene_quality import extract_brouhaha_frames

        brouhaha_frames = extract_brouhaha_frames([audio])[0]
    # L1 measurements are already the raw dB this fit needs — no ``calibration`` argument, since
    # fitting anchors from values that had anchors applied would be circular.
    rows = harvest_quality_measurements(audio=audio, brouhaha=brouhaha_frames, grid=BucketGrid(0.5, 0.5))

    def _median(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    # Brouhaha's SNR when available, else the spectral-gating estimator. Named explicitly rather
    # than averaging: the estimators use different noise-floor definitions, so a mean of them is
    # not an estimate of any one quantity.
    snr = _median("snr_brouhaha_db")
    if snr is None:
        snr = _median("snr_spectral_gating_db")
    return {"snr_db": snr, "c50_db": _median("c50_brouhaha_db")}


def _fit_anchors(
    true_db: list[float], est_db: list[float], clean_anchor: float, floor_anchor: float
) -> dict[str, float]:
    """Least-squares line est ≈ a·true + b → anchors = predicted est at the true targets."""
    import numpy as np

    a, b = np.polyfit(np.asarray(true_db, dtype=float), np.asarray(est_db, dtype=float), 1)
    clean_db = float(a * clean_anchor + b)
    floor_db = float(a * floor_anchor + b)
    if clean_db <= floor_db:  # estimator not monotone enough to fit — fail loudly
        raise ValueError(
            f"fitted anchors are inverted (clean {clean_db:.2f} <= floor {floor_db:.2f}); "
            f"estimator slope a={a:.3f} — check the sweep / estimator"
        )
    return {"clean_db": round(clean_db, 3), "floor_db": round(floor_db, 3), "slope": round(float(a), 4)}


def main(argv: list[str] | None = None) -> int:
    """Run the sweeps, fit, validate, persist."""
    args = parse_args(argv)
    import numpy as np

    from senselab.audio.workflows.audio_analysis.calibration import (
        DEFAULT_PROFILE,
        linear_db_to_unit,
        validate_profile,
    )

    rng = np.random.default_rng(args.seed)

    # Clean reference via the pipeline's own loader chain (16 kHz mono parity).
    from senselab.audio.tasks.input_output import read_audios
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    audio = read_audios([str(args.audio)])[0]
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], resample_rate=16000)[0]
    speech = audio.waveform.detach().cpu().numpy().squeeze().astype("float32")
    sr = 16000

    # ── SNR sweep ────────────────────────────────────────────────────────
    snr_true, snr_est = [], []
    noise = _synth_noise(len(speech), args.noise, rng)
    for target_db in args.snr_sweep_db:
        est = _estimate(_mix_at_snr(speech, noise, target_db), sr, brouhaha=False)
        if est["snr_db"] is not None:
            snr_true.append(float(target_db))
            snr_est.append(float(est["snr_db"]))
        print(f"  SNR sweep: true={target_db:6.1f} dB → estimated={est['snr_db']}")
    if len(snr_true) < 3:
        print("ERROR: fewer than 3 usable SNR points — cannot fit anchors", file=sys.stderr)
        return 2
    snr_fit = _fit_anchors(snr_true, snr_est, args.clean_anchor_db, args.floor_anchor_db)

    # ── RT60 / C50 sweep (Brouhaha only — the DSP path has no reverb estimator) ──
    c50_fit: dict[str, float] | None = None
    c50_true, c50_est = [], []
    if not args.no_brouhaha:
        for rt60 in args.rt60_sweep_s:
            if rt60 <= 0:
                reverbed, true_c50 = speech, 40.0  # effectively anechoic
            else:
                h = _exponential_rir(rt60, sr, rng)
                reverbed = np.convolve(speech, h)[: len(speech)].astype("float32")
                peak = max(1e-9, float(np.max(np.abs(reverbed))))
                reverbed = reverbed / peak if peak > 1.0 else reverbed
                true_c50 = _true_c50_db(h, sr)
            est = _estimate(reverbed, sr, brouhaha=True)
            print(f"  RT60 sweep: rt60={rt60:4.1f} s (true C50≈{true_c50:6.1f} dB) → estimated={est['c50_db']}")
            if est["c50_db"] is not None:
                c50_true.append(float(true_c50))
                c50_est.append(float(est["c50_db"]))
        if len(c50_true) >= 3:
            c50_fit = _fit_anchors(c50_true, c50_est, clean_anchor=30.0, floor_anchor=-5.0)
        else:
            print("warn: Brouhaha C50 unavailable — keeping default reverb anchors", file=sys.stderr)

    # ── Profile (data-model §5) + provenance ────────────────────────────
    profile: dict[str, Any] = {
        "version": "1",
        "snr": {"type": "linear_db_to_unit", "clean_db": snr_fit["clean_db"], "floor_db": snr_fit["floor_db"]},
        "reverb_c50": (
            {"type": "linear_db_to_unit", "clean_db": c50_fit["clean_db"], "floor_db": c50_fit["floor_db"]}
            if c50_fit
            else dict(DEFAULT_PROFILE["reverb_c50"])
        ),
        "bandwidth": dict(DEFAULT_PROFILE["bandwidth"]),
        "temperature": {"speech_presence": args.temperature_speech_presence, "asr": args.temperature_asr},
        "provenance": {
            "fitted_by": "scripts/calibrate_scene_quality.py",
            "audio": args.audio.name,
            "noise": args.noise,
            "snr_sweep_db": list(args.snr_sweep_db),
            "rt60_sweep_s": list(args.rt60_sweep_s),
            "seed": args.seed,
            "snr_fit": snr_fit,
            "c50_fit": c50_fit or "defaults (Brouhaha unavailable or --no-brouhaha)",
            "temperature_note": "CLI passthrough — fit requires labeled correctness (adaptive eval harness)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    validate_profile(profile, source="fitted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # indent=4 + trailing newline to match the repo's pretty-format-json /
    # end-of-file-fixer hooks, so a profile fitted into the package tree is
    # committable without a reformat round-trip.
    args.out.write_text(json.dumps(profile, indent=4) + "\n")
    print(f"profile: {args.out}")

    # ── Validation table + plot (T038 / FR-022, SC-007) ──────────────────
    reported = [linear_db_to_unit(e, snr_fit["clean_db"], snr_fit["floor_db"]) for e in snr_est]
    monotone = all(reported[i] <= reported[i + 1] + 1e-9 for i in range(len(reported) - 1))
    table = {
        "snr": [
            {"true_db": t, "estimated_db": e, "reported_degradation": round(r, 4)}
            for t, e, r in zip(snr_true, snr_est, reported)
        ],
        "c50": [{"true_db": t, "estimated_db": e} for t, e in zip(c50_true, c50_est)],
        "monotone_reported_vs_true": monotone,
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(table, indent=4) + "\n")
    print(f"table:   {args.table}  (monotone={monotone})")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(snr_true, snr_est, "o-", label="estimated dB")
        ax1.axhline(snr_fit["clean_db"], ls="--", c="green", label=f"clean anchor {snr_fit['clean_db']:.1f}")
        ax1.axhline(snr_fit["floor_db"], ls="--", c="red", label=f"floor anchor {snr_fit['floor_db']:.1f}")
        ax1.set_xlabel("true SNR (dB)")
        ax1.set_ylabel("estimated SNR (dB)")
        ax1.set_title("Estimator vs truth (SNR sweep)")
        ax1.legend(fontsize=8)
        ax2.plot(snr_true, reported, "s-", color="tab:red")
        ax2.set_xlabel("true SNR (dB)")
        ax2.set_ylabel("reported degradation [0,1]")
        ax2.set_title(f"Calibrated mapping (monotone={monotone})")
        ax2.invert_xaxis()
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print(f"plot:    {args.plot}")
    except Exception as exc:  # noqa: BLE001 — plot is a best-effort sidecar
        print(f"warn: validation plot failed: {exc!r}", file=sys.stderr)

    if not monotone:
        print("ERROR: reported degradation is not monotone vs true SNR (SC-007)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
