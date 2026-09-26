"""Per-family tables over the census rows: the three rates, and the evidence behind every False."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

IN_FAMILY = (
    "maximum-phonation-time",
    "maximum-phonation-time-v2",
    "prolonged-vowel",
    "glides-low-to-high",
    "glides-high-to-low",
    "high-to-low",
)


def load(path: Path) -> list[dict]:
    """Every census row that parsed."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    """Print the tables."""
    rows = load(Path(sys.argv[1]))
    ok = [r for r in rows if r.get("ok")]
    print(f"rows={len(rows)} ok={len(ok)} failed={len(rows) - len(ok)}")
    fams = sorted({r["family"] for r in ok}, key=lambda f: (f not in IN_FAMILY, f))

    print("\n== (1)(2)(3) per family ==")
    hdr = (
        f"{'family':<30}{'n':>4}{'routed':>7}{'rtd+abs':>8}{'rate':>7}"
        f"{'claimNF':>8}{'rate':>7}{'True':>6}{'False':>6}{'UNDET':>6}{'noRep':>6}"
    )
    print(hdr)
    for fam in fams:
        sub = [r for r in ok if r["family"] == fam]
        n = len(sub)
        routed = [r for r in sub if (r.get("routes") or {}).get("VOICE") == "routed"]
        rtd_abs = [r for r in routed if (r.get("findings") or {}).get("VOICE") == "absent"]
        claim_nf = [r for r in sub if (r.get("hints") or {}).get("VOICE") == "claimed_not_found"]
        conf = collections.Counter(str(r.get("voice_conformance")) for r in sub)
        print(
            f"{fam:<30}{n:>4}{len(routed):>7}{len(rtd_abs):>8}"
            f"{(len(rtd_abs) / max(len(routed), 1)):>7.2f}"
            f"{len(claim_nf):>8}{(len(claim_nf) / n):>7.2f}"
            f"{conf['True']:>6}{conf['False']:>6}{conf['UNDETERMINED']:>6}{conf['None']:>6}"
        )

    print("\n== (4) evidence behind every VOICE False and every absent ==")
    for fam in fams:
        sub = [r for r in ok if r["family"] == fam]
        bad = [
            r for r in sub if r.get("voice_conformance") is False or (r.get("findings") or {}).get("VOICE") == "absent"
        ]
        if not bad:
            continue
        amp = collections.Counter()
        gate = collections.Counter()
        no_amp = 0
        for r in bad:
            amp[min(r["amplitude_spans_n"], 10)] += 1
            if r["amplitude_spans_n"] == 0:
                no_amp += 1
            long = [c for c in r["census"] if (c.get("dur_s") or 0) >= 0.5]
            if not long:
                gate["no_span_over_production_min_s"] += 1
                continue
            worst = max(long, key=lambda c: c.get("dur_s") or 0)
            gate[str(worst.get("rejected_by"))] += 1
        print(f"\n-- {fam}: {len(bad)} of {len(sub)} recordings")
        print(
            f"   tracks present on {sum(1 for r in bad if r['tracks_present'])}/{len(bad)};"
            f" continuity on {sum(1 for r in bad if r['continuity_present'])}/{len(bad)};"
            f" zero amplitude spans on {no_amp}/{len(bad)}"
        )
        print("   gate that rejected the LONGEST qualifying-length carrier:")
        for name, count in gate.most_common():
            print(f"     {name:<36} {count:>4}  ({count / len(bad):.2f})")

    print("\n== longest carrier: duration vs. measured F0 spread, in-family sustained ==")
    print(f"{'family':<28}{'n':>4}{'med_dur':>9}{'med_spread':>11}{'>2st':>6}{'>12st':>7}{'med_vf':>8}{'med_stat':>9}")
    for fam in fams:
        sub = [r for r in ok if r["family"] == fam and r.get("pattern") in ("sustained", "detect")]
        longs = []
        for r in sub:
            cand = [c for c in r["census"] if c.get("f0_spread_semitones") is not None]
            if cand:
                longs.append(max(cand, key=lambda c: c.get("dur_s") or 0))
        if not longs:
            continue
        med = lambda key: sorted(c[key] for c in longs if c.get(key) is not None)[len(longs) // 2]  # noqa: E731
        over2 = sum(1 for c in longs if (c.get("f0_spread_semitones") or 0) > 2.0)
        over12 = sum(1 for c in longs if (c.get("f0_spread_semitones") or 0) > 12.0)
        print(
            f"{fam:<28}{len(longs):>4}{med('dur_s'):>9.2f}{med('f0_spread_semitones'):>11.2f}"
            f"{over2:>6}{over12:>7}{med('voiced_fraction'):>8.2f}{med('stationarity'):>9.3f}"
        )

    print("\n== duration of the carrier vs. whether the spread gate rejected it (sustained, in family) ==")
    buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
    tally = {b: [0, 0] for b in buckets}
    for r in ok:
        if r["family"] not in IN_FAMILY or r.get("pattern") != "sustained":
            continue
        for c in r["census"]:
            if c.get("f0_spread_semitones") is None:
                continue
            d = c.get("dur_s") or 0
            for b in buckets:
                if b[0] <= d < b[1]:
                    tally[b][1] += 1
                    if c.get("rejected_by") == "f0_spread_max_semitones":
                        tally[b][0] += 1
    print(f"{'carrier duration (s)':<24}{'rejected':>10}{'n':>6}{'rate':>8}")
    for b in buckets:
        bad, n = tally[b]
        if n:
            print(f"{f'{b[0]}-{b[1]}':<24}{bad:>10}{n:>6}{bad / n:>8.2f}")

    print("\n== carriers the spread gate rejected: is the production steady, or is the tracker? ==")
    print(
        f"{'family':<30}{'n':>4}{'med_windows':>13}{'med_spread_med':>16}{'med_spread_max':>16}"
        f"{'med_frac_over':>15}{'max~octave':>12}{'f0_range_hz':>14}"
    )
    for fam in fams:
        rej = []
        rng = []
        for r in ok:
            if r["family"] != fam:
                continue
            p_ = r.get("tracks_params") or {}
            if p_.get("f0_min_hz") is not None:
                rng.append((p_["f0_min_hz"], p_["f0_max_hz"]))
            rej += [c for c in r["census"] if c.get("rejected_by") == "f0_spread_max_semitones" and c.get("windows_n")]
        if not rej:
            continue
        srt = lambda key: sorted(c[key] for c in rej)  # noqa: E731
        mid = len(rej) // 2
        frac = sorted(c["windows_over_max"] / c["windows_n"] for c in rej)
        octave = sum(1 for c in rej if 11.0 <= c["f0_spread_semitones"] <= 13.0 or c["f0_spread_semitones"] >= 23.0)
        band = f"{rng[len(rng) // 2][0]:.0f}-{rng[len(rng) // 2][1]:.0f}" if rng else "-"
        print(
            f"{fam:<30}{len(rej):>4}{srt('windows_n')[mid]:>13}{srt('spread_median')[mid]:>16.2f}"
            f"{srt('f0_spread_semitones')[mid]:>16.2f}{frac[mid]:>15.3f}{octave:>12}{band:>14}"
        )

    print("\n== the decisive classification: what stood between the recording and a carrier ==")
    print(
        f"{'family':<30}{'no_carrier':>11}{'no_span>=.5s':>13}{'none_voiced':>12}"
        f"{'voiced_but_spread':>19}{'voiced_but_cont':>17}{'voiced_but_lex':>16}{'glide_dom':>11}"
    )
    for fam in fams:
        sub = [r for r in ok if r["family"] == fam and r["carriers_n"] == 0]
        if not sub:
            continue
        counts = collections.Counter()
        for r in sub:
            long = [c for c in r["census"] if (c.get("dur_s") or 0) >= 0.5]
            if not long:
                counts["no_span"] += 1
                continue
            voiced = [c for c in long if (c.get("voiced_fraction") or 0) >= 0.5]
            lex = [c for c in long if c.get("rejected_by") == "lexical_separator"]
            if not voiced:
                counts["lex" if lex and not voiced else "none_voiced"] += 1
                continue
            by = {c.get("rejected_by") for c in voiced}
            if "f0_spread_max_semitones" in by:
                counts["spread"] += 1
            elif "continuity_min" in by:
                counts["cont"] += 1
            elif "dominant_segment_min_fraction" in by or "longest_monotone_run_none" in by:
                counts["glide_dom"] += 1
            elif "lexical_separator" in by:
                counts["lex"] += 1
            else:
                counts["other"] += 1
        print(
            f"{fam:<30}{len(sub):>11}{counts['no_span']:>13}{counts['none_voiced']:>12}"
            f"{counts['spread']:>19}{counts['cont']:>17}{counts['lex']:>16}{counts['glide_dom']:>11}"
        )

    print("\n== why each in-family False was False: no carrier, or the count-in ==")
    print(f"{'family':<30}{'False':>7}{'no_carrier':>12}{'carrier+countin':>17}{'lexwords=0':>12}")
    for fam in [f for f in fams if f in IN_FAMILY]:
        sub = [r for r in ok if r["family"] == fam and r.get("voice_conformance") is False]
        if not sub:
            continue
        no_carrier = [r for r in sub if r["carriers_n"] == 0]
        with_carrier = [r for r in sub if r["carriers_n"] > 0]
        zero_lex = [r for r in with_carrier if r.get("lexical_words_n") == 0]
        print(f"{fam:<30}{len(sub):>7}{len(no_carrier):>12}{len(with_carrier):>17}{len(zero_lex):>12}")

    print("\n== does VOICE record why a carrier was rejected? spans it proposed on a False ==")
    for fam in [f for f in fams if f in IN_FAMILY]:
        sub = [r for r in ok if r["family"] == fam and r.get("voice_conformance") is False]
        if not sub:
            continue
        spans = collections.Counter((r.get("voice_report") or {}).get("spans_n") for r in sub)
        devs = collections.Counter(tuple(sorted(r.get("voice_deviations") or ())) for r in sub)
        print(f"  {fam}: spans_n {dict(spans)}; deviation sets {dict(list(devs.most_common(4)))}")

    print("\n== flag records per recording, and their grounds ==")
    voice_reasons = collections.Counter()
    per_rec = collections.Counter()
    for r in ok:
        vs = [w for node, outcome, w in (r.get("reasons") or []) if node == "VOICE" and outcome == "flag"]
        per_rec[len(vs)] += 1
        for w in vs:
            voice_reasons[w.replace(r.get("declared_family") or "\x00", "<family>")] += 1
    print("VOICE flag records per recording:", dict(sorted(per_rec.items())))
    for why, count in voice_reasons.most_common(12):
        print(f"  {count:>4}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
