"""Emit a versioned detection-margin profile from measured classifier verdicts (T070, FR-022).

The margin ladder decides which background sources a run is willing to report, so it must
be auditable rather than tuned by hand. This script rebuilds a profile's ``machine_basis``
from the level-probe verdicts actually measured on this host and refuses to emit a profile
whose evidence does not hold up.

Two refusals, both hard errors rather than warnings:

**An unmarked provisional figure.** A number whose status is ``provisional`` must carry a
note saying what is unverified about it. Without the note the profile reads as settled, and
the whole point of the derivation block is that a reader can tell settled from not.

**A confident tier the classifier cannot actually reach.** The ladder is only defensible if
the human criterion and the measured machine floor are satisfied at once (SC-017). If the
probe reports the classifier needs more headroom than the confident tier grants, emitting
the profile anyway would publish a threshold known not to work.

Usage::

    uv run python scripts/calibrate_detection_margin.py \\
        --level-verdicts artifacts/level_probe/level-verdicts.json \\
        --out src/senselab/audio/workflows/audio_analysis/data/detection_margin/<version>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from senselab.audio.workflows.audio_analysis.calibration import (  # noqa: E402
    load_detection_margin_profile,
    validate_detection_margin_profile,
)

DEFAULT_BASE = REPO_ROOT / "src/senselab/audio/workflows/audio_analysis/data/detection_margin/2026-07-29.json"


def machine_basis_from_verdicts(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn measured level-probe verdicts into ``derivation.machine_basis`` claims.

    One claim per classifier, naming the SNR at which it became reliable. A classifier the
    probe skipped contributes nothing rather than a claim with a missing number — an absent
    measurement and a measurement of zero license opposite conclusions.
    """
    claims: list[dict[str, Any]] = []
    for verdict in doc.get("verdicts") or []:
        if not isinstance(verdict, dict):
            continue
        name = str(verdict.get("classifier") or verdict.get("model") or "unknown")
        floor = verdict.get("reliable_floor_snr_db", verdict.get("reliable_floor_db"))
        if floor is None:
            continue
        claims.append(
            {
                "claim": f"{name} reliable floor {float(floor):.1f} dB SNR",
                "source": "level probe (measured)",
                "status": "verified",
                "measured_on": doc.get("host") or doc.get("generated_on") or "unrecorded",
            }
        )
    return claims


def check_confident_tier_is_reachable(profile: dict[str, Any], claims: list[dict[str, Any]]) -> None:
    """Refuse a confident tier no measured classifier can reach.

    Raises:
        ValueError: If every measured floor sits above the confident margin. Publishing that
            profile would ship a threshold already known not to work on this host.
    """
    confident = float(profile["margins_db"]["confident"])
    floors = []
    for claim in claims:
        text = str(claim.get("claim", ""))
        for token in text.split():
            try:
                floors.append(float(token))
            except ValueError:
                continue
    if floors and min(floors) > confident:
        raise ValueError(
            f"confident tier is {confident} dB but the lowest measured classifier floor is "
            f"{min(floors)} dB SNR — no classifier reaches this tier on the probed host (SC-017)"
        )


def build_profile(base: dict[str, Any], verdicts: dict[str, Any], *, calibrated_as: str) -> dict[str, Any]:
    """Return the base profile with its machine basis replaced by measured claims.

    ``profile_version`` is the *schema* version and is carried over untouched — the loader
    pins it, and a profile stamped with its own name there would be rejected at read time.
    The profile's identity lives in ``calibrated_as`` and in its filename.
    """
    out = json.loads(json.dumps(base))  # deep copy without importing copy for one call
    out["calibrated_as"] = calibrated_as
    claims = machine_basis_from_verdicts(verdicts)
    if not claims:
        raise ValueError(
            "no measured verdicts carry a reliable floor — a profile emitted from this would "
            "claim a machine basis it does not have"
        )
    out.setdefault("derivation", {})["machine_basis"] = claims
    check_confident_tier_is_reachable(out, claims)
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level-verdicts", type=Path, required=True, help="level-verdicts.json from the probe")
    parser.add_argument("--out", type=Path, required=True, help="destination profile JSON")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="profile to take non-measured fields from")
    parser.add_argument("--name", default=None, help="calibration name to record (defaults to the out stem)")
    args = parser.parse_args(argv)

    try:
        verdicts = json.loads(args.level_verdicts.read_text())
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {args.level_verdicts}: {exc}", file=sys.stderr)
        return 2

    try:
        # The base is validated on load, so a base whose own derivation is unsound — an
        # unmarked provisional figure, say — is refused here rather than propagated into a
        # freshly-stamped profile that would look newly calibrated.
        base = load_detection_margin_profile(args.base)
        profile = build_profile(base, verdicts, calibrated_as=args.name or args.out.stem)
        # The same validator the loader applies, so a profile that would be rejected at read
        # time is never written in the first place.
        validate_detection_margin_profile(profile, source=str(args.out))
    except (ValueError, OSError) as exc:
        print(f"ERROR: refusing to emit profile: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=1) + "\n")
    print(f"wrote {args.out} ({len(profile['derivation']['machine_basis'])} measured claim(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
