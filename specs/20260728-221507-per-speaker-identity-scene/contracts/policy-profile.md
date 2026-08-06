# Contract: policy profiles

**Files**:
`src/senselab/audio/workflows/audio_analysis/data/detection_margin/<version>.json`,
`src/senselab/audio/workflows/audio_analysis/adaptive/policy/default.yaml` (extended)

Constitution principle VIII applies with full force here: **no threshold in this feature
may appear as a literal in code**. Every value below is a profile entry with a default, so
the common case needs zero configuration while remaining tunable.

## `detection_margin/<version>.json`

```json
{
  "profile_version": "detection-margin/2026-07-29",

  "margins_db": {
    "reject_below": 3.0,
    "candidate": 3.0,
    "probable": 6.0,
    "confident": 10.0
  },

  "level": {
    "target_lufs": -23.0,
    "true_peak_ceiling_dbtp": -1.0,
    "gain_cap_db": 10.0,
    "reject_below_pregain_dbfs": -45.0,
    "stable_band_dbfs": [-35.0, -15.0]
  },

  "noise_floor": {
    "quantile": 0.10,
    "window_s": 20.0,
    "max_iterations": 3,
    "event_exclusion_db": 6.0,
    "band_smoothing_bins": 5,
    "top_percentile_cap": 0.95,
    "condition_on_target_activity": true,
    "freeze_inside_events": true,
    "recorder_margin_db": 3.0
  },

  "guards": {
    "flatness_max": 0.30,
    "min_occupancy": 0.40,
    "min_duration_s": {"default": 0.20},
    "hysteresis": {"trigger_tier": "confident", "extend_tier": "probable"},
    "min_distance_separation_db": 6.0,
    "quarantined_labels": [
      "White noise", "Pink noise", "Noise", "Static", "Environmental noise",
      "Hum", "Mains hum", "Hiss", "Throbbing", "Waterfall", "Water",
      "Gurgling", "Spray", "Sine wave", "Silence", "Inside, small room"
    ]
  },

  "mask": {
    "guard_interval_s": 0.50,
    "negligible_fraction": 0.05,
    "min_region_s": 1.0,
    "max_padding_fraction": 0.50,
    "target_event_types_by_task": {
      "speech": ["speech", "breath", "mouth_noise"],
      "breath": ["breath"],
      "cough": ["cough", "throat_clear"]
    },
    "fallback_target_event_types": ["speech", "breath", "cough", "mouth_noise"]
  },

  "derivation": {
    "human_basis": [
      {"claim": "minimum measurability ~+3 dB", "source": "ISO 1996-2:2017", "status": "verified"},
      {"claim": "10 dB octave / 13 dB third-octave above masked threshold", "source": "ISO 7731", "status": "verified"},
      {"claim": "+5 adverse / +10 significant over background", "source": "BS 4142", "status": "verified"},
      {"claim": "TNR >= 8 dB, PR >= 9 dB for tone prominence", "source": "ECMA-74 17th ed.", "status": "verified"},
      {"claim": "partial-masking transition -3 to +15 dB", "source": "Moore/Glasberg partial loudness", "status": "provisional",
       "note": "surfaced in indexed text; primary source paywalled. Corroborates but does not set the value."}
    ],
    "machine_basis": [
      {"claim": "short-window reliable floor 5-10 dB SNR", "source": "level probe (measured)", "status": "verified"},
      {"claim": "long-window reliable floor 15-20 dB SNR non-speech", "source": "level probe (measured)", "status": "verified"},
      {"claim": "noise-family label contamination from ~20 dB SNR", "source": "level probe (measured)", "status": "verified"},
      {"claim": "short-window silence floor ~-60 dBFS, learned not arithmetic", "source": "level probe (measured)", "status": "verified"}
    ],
    "agreement_note": "Human criteria place confident identification near +10 dB; the short-window classifier's measured reliable floor is 5-10 dB SNR. The +10 dB confident tier satisfies both simultaneously (SC-017).",
    "derived_statistics_status": "provisional",
    "derived_statistics_note": "The exponential-periodogram bias correction 1/(-ln(1-q)), the per-bin sigma of 5.57 dB, and the patch-variance collapse are straightforward chi-squared results but were not found stated in this form in the literature. Validate on synthetic noise before relying on them (research.md open risk 2)."
  }
}
```

### Invariants

- `margins_db` is monotone: `reject_below <= candidate <= probable <= confident`.
- `noise_floor.quantile` in `(0, 0.5]`; the bias correction is computed from it, never
  stored independently, so the two cannot drift apart.
- `level.gain_cap_db <= 10.0` — the measured clipping inflection. Raising it requires a
  new derivation entry.
- `derivation.human_basis` and `derivation.machine_basis` are both non-empty, and
  `agreement_note` is present (FR-022, SC-017).
- Every derivation claim carries `status` of `verified` or `provisional`; a `provisional`
  claim requires a `note` (FR-022). The calibration script hard-errors otherwise.
- A profile whose only support for a margin value is `provisional` is rejected — a
  provisional figure may corroborate a value but may not set it.
- `mask.target_event_types_by_task` keys are the accepted `--task-type` values; anything
  else triggers the fallback and is recorded as such (FR-033b).

### Thresholds that quantify otherwise-vague spec language

These three exist so that FR-021b, FR-020/SC-012, and FR-043 have somewhere to live rather
than becoming literals in code (constitution VIII). Unlike the 3/6/10 dB ladder, these
values are **starting points, not derived** — they carry no corroboration and should be
revised freely on evidence.

| Key | Quantifies | Effect |
|---|---|---|
| `noise_floor.recorder_margin_db` | FR-021b's "within a few dB" of the recorder floor | A band whose floor is within this margin of the estimated capture-chain self-noise yields `binding_floor: "recorder"`, and no perceptual claim is made for findings in it |
| `guards.min_distance_separation_db` | "materially different distances" in FR-020 and SC-012 | Two same-type sources whose levels differ by at least this much are treated as being at materially different distances for SC-012's purposes |
| `mask.min_region_s`, `mask.max_padding_fraction` | FR-043's "materially shorter than that window" | A mask region shorter than `min_region_s`, or whose excised segment would exceed `max_padding_fraction` padding, sets `supports_long_window: false` (FR-045) |

- All three MUST be read from the profile. A literal in code is a constitution VIII
  violation even when the value happens to match the default.

## `adaptive/policy/default.yaml` — added keys

```yaml
influence:
  max_rounds: 5
  oscillation_window: 3          # rounds inspected for alternation (FR-011e)
  min_uncertainty_improvement: 0.01   # below this, treat as no progress
  uncertainty_gate:
    # effective_weight = base_weight * (1 - uncertainty) ** exponent
    exponent: 1.0
  derivation_gate:
    independent: 1.0
    derived: 0.4                 # < 1.0 is mandatory (FR-011c)
  weights:
    identity_to_diarization: 1.0
    diarization_to_speaker_assignment: 1.0
    background_mask_to_presence: 0.6
    utterance_to_speaker_attribution: 0.5

speaker_count:
  multimodal_threshold: 0.15     # a count above this counts as a supported mode
  agreement_mass_for_single: 0.90   # SC-001 gate
```

### Invariants

- `derivation_gate.derived < derivation_gate.independent` (FR-011c, SC-030).
- `max_rounds` bounds iteration; reaching it sets `termination_reason == "budget"` and
  `converged == false` rather than presenting the last state as settled (FR-011h).
- `oscillation_window >= 2` — alternation cannot be detected with a single round.
- All keys have defaults; a run with no policy override is valid and reproducible.
- Changing any value changes `policy_hash`, which appears in `final/transcript.json` and
  the convergence report, so a result is always attributable to the policy that produced
  it.
