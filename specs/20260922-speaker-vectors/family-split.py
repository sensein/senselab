"""Does the airway family explain the near-orthogonal tail, and is `voice` worth keeping?

Re-analysis of the per-extent vectors ``exp_c.py`` wrote for D-7. No embedding is re-run: every
vector here came off the production window grid and aggregator, one vector per admitted task
extent, carrying that extent's ``family``.

Two questions, two sections.

A. The tail. Per-extent cosine to the subject's own leave-one-out centroid, split by family.
B. The three-way separation comparison -- speech, speech+voice, all three -- unmatched and
   supply-matched.

Usage: ``family_split.py <exp_c_dir>``
"""

import glob
import json
import sys

import numpy as np

RNG = np.random.default_rng(20260922)
FAMILY_SETS = {
    "speech": ("speech",),
    "speech+voice": ("speech", "voice"),
    "all-three": ("speech", "voice", "airway"),
}


def unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise along the last axis, leaving a zero vector alone.

    Args:
        v: Any real array; the last axis is the vector axis.

    Returns:
        The normalised array.
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def auc_of(same: np.ndarray, diff: np.ndarray) -> float:
    """Rank-based AUC of same-speaker scores against between-speaker scores.

    Args:
        same: Within-speaker cosines.
        diff: Between-speaker cosines.

    Returns:
        The area under the ROC curve.
    """
    allv = np.concatenate([same, diff])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[: len(same)].sum() - len(same) * (len(same) + 1) / 2) / (len(same) * len(diff)))


def eer_of(same: np.ndarray, diff: np.ndarray) -> float:
    """Equal error rate, by one sort rather than a threshold sweep.

    Args:
        same: Within-speaker cosines.
        diff: Between-speaker cosines.

    Returns:
        The equal error rate.
    """
    ts = np.unique(np.concatenate([same, diff]))
    ss, dd = np.sort(same), np.sort(diff)
    far = 1.0 - np.searchsorted(dd, ts, side="left") / len(dd)
    frr = np.searchsorted(ss, ts, side="left") / len(ss)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


def load(root: str) -> dict:
    """Load every shard's per-extent vectors into one subject-keyed mapping.

    Args:
        root: The directory holding ``extent_vectors-*.json``.

    Returns:
        ``{subject: {"vectors": [...], "meta": [...]}}``.
    """
    data: dict = {}
    for p in sorted(glob.glob(root + "/extent_vectors-*.json")):
        with open(p) as fh:
            data.update(json.load(fh))
    return data


def score(subs: list, halves: dict) -> dict:
    """Score one condition: within is the diagonal, between is everything else.

    Args:
        subs: The subjects in this condition, in a fixed order.
        halves: ``{subject: {"a": vector, "b": vector}}``.

    Returns:
        The metric mapping D-7's table uses.
    """
    a = np.stack([halves[s]["a"] for s in subs])
    b = np.stack([halves[s]["b"] for s in subs])
    m = a @ b.T
    same = np.diag(m).copy()
    diff = m[~np.eye(len(subs), dtype=bool)]
    dprime = (same.mean() - diff.mean()) / np.sqrt(0.5 * (same.var() + diff.var()))
    return {
        "n_subjects": len(subs),
        "within_mean": same.mean(),
        "within_p05": np.percentile(same, 5),
        "between_mean": diff.mean(),
        "between_p95": np.percentile(diff, 95),
        "auc": auc_of(same, diff),
        "eer": eer_of(same, diff),
        "dprime": dprime,
        "rank1": float((m.argmax(axis=1) == np.arange(len(subs))).mean()),
    }


def table(rows: list, title: str) -> None:
    """Print one comparison in the columns design.md D-7 already uses.

    Args:
        rows: ``(condition name, metrics, mean extents per subject)`` triples.
        title: The table's heading.
    """
    print(f"\n== {title} ==")
    hdr = (
        f"{'condition':>16} {'subj':>5} {'ext/subj':>9} {'w.mean':>8} {'w.p05':>8} {'b.mean':>8} "
        f"{'b.p95':>8} {'AUC':>8} {'EER':>8} {'dprime':>7} {'rank-1':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, r, per in rows:
        print(
            f"{name:>16} {r['n_subjects']:5d} {per:9.2f} {r['within_mean']:8.4f} {r['within_p05']:8.4f} "
            f"{r['between_mean']:8.4f} {r['between_p95']:8.4f} {r['auc']:8.5f} {r['eer']:8.5f} "
            f"{r['dprime']:7.3f} {r['rank1']:7.4f}"
        )


def section_a(data: dict) -> None:
    """Per-extent cosine to the subject's own leave-one-out centroid, split by family.

    Args:
        data: The loaded per-extent vectors.
    """
    fam_counts: dict = {}
    for s in data:
        for m in data[s]["meta"]:
            fam_counts[m["family"]] = fam_counts.get(m["family"], 0) + 1
    print(f"extent vectors by family: {dict(sorted(fam_counts.items(), key=lambda kv: -kv[1]))}")

    by_family: dict = {}
    dur_by_family: dict = {}
    for s in sorted(data):
        vectors = unit(np.asarray(data[s]["vectors"], dtype=np.float64))
        meta = data[s]["meta"]
        if len(vectors) < 3:
            continue
        total = vectors.sum(axis=0)
        for i in range(len(vectors)):
            loo = unit(total - vectors[i])
            by_family.setdefault(meta[i]["family"], []).append(float(vectors[i] @ loo))
            dur_by_family.setdefault(meta[i]["family"], []).append(meta[i]["duration_s"])

    print("\n== A. per-extent cosine to the subject's own leave-one-out centroid, by family ==")
    hdr = (
        f"{'family':>10} {'n':>7} {'mean':>8} {'p05':>8} {'p25':>8} {'median':>8} {'p75':>8} "
        f"{'<0.1':>7} {'<0.2':>7} {'med dur s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    pooled = []
    for fam in ("speech", "voice", "airway"):
        if fam not in by_family:
            continue
        a = np.asarray(by_family[fam])
        d = np.asarray(dur_by_family[fam])
        pooled.append(a)
        print(
            f"{fam:>10} {len(a):7d} {a.mean():8.4f} {np.percentile(a, 5):8.4f} {np.percentile(a, 25):8.4f} "
            f"{np.median(a):8.4f} {np.percentile(a, 75):8.4f} {float((a < 0.1).mean()):7.3f} "
            f"{float((a < 0.2).mean()):7.3f} {np.median(d):10.2f}"
        )
    allc = np.concatenate(pooled)
    print(
        f"{'ALL':>10} {len(allc):7d} {allc.mean():8.4f} {np.percentile(allc, 5):8.4f} "
        f"{np.percentile(allc, 25):8.4f} {np.median(allc):8.4f} {np.percentile(allc, 75):8.4f} "
        f"{float((allc < 0.1).mean()):7.3f} {float((allc < 0.2).mean()):7.3f}"
    )

    cut = np.percentile(allc, 5)
    print(f"\nbottom 5% of all extents is cos < {cut:.4f}. Composition of that tail:")
    tail = [
        (fam, int((np.asarray(by_family[fam]) < cut).sum()), len(by_family[fam]))
        for fam in ("speech", "voice", "airway")
        if fam in by_family
    ]
    tot_tail = max(sum(t[1] for t in tail), 1)
    tot_all = sum(t[2] for t in tail)
    for fam, ntail, nall in tail:
        share, base = ntail / tot_tail, nall / tot_all
        print(
            f"  {fam:>8}: {ntail:6d} of {tot_tail} tail extents ({100 * share:5.1f}%) "
            f"vs {100 * base:5.1f}% of the corpus  -> enrichment {share / base:.2f}x"
        )


def section_b(data: dict) -> None:
    """The three-way separation comparison, unmatched and supply-matched.

    Args:
        data: The loaded per-extent vectors.
    """
    subs_all = sorted(data)

    def fam_idx(sub: str, allowed: tuple) -> list:
        """Positions of this subject's extents whose family is in ``allowed``."""
        return [i for i, m in enumerate(data[sub]["meta"]) if m["family"] in allowed]

    def build(subs: list, picker) -> tuple:  # noqa: ANN001
        """Pool each subject's picked extents into two disjoint halves."""
        halves: dict = {}
        per = []
        for sub in subs:
            idx = picker(sub)
            if idx is None or len(idx) < 4:
                continue
            vectors = unit(np.asarray(data[sub]["vectors"], dtype=np.float64))[idx]
            order = RNG.permutation(len(vectors))
            a_idx, b_idx = order[: len(vectors) // 2], order[len(vectors) // 2 : 2 * (len(vectors) // 2)]
            halves[sub] = {"a": unit(vectors[a_idx].mean(axis=0)), "b": unit(vectors[b_idx].mean(axis=0))}
            per.append(len(vectors))
        return sorted(halves), halves, float(np.mean(per)) if per else 0.0

    rows = []
    for name, allowed in FAMILY_SETS.items():
        kept, halves, per = build(subs_all, lambda s, a=allowed: fam_idx(s, a))
        rows.append((name, score(kept, halves), per))
    table(rows, "B1. unmatched: every extent each condition admits (supply differs by condition)")

    speech_n = {s: len(fam_idx(s, ("speech",))) for s in subs_all}
    print(f"\nsupply-matched cohort base: {sum(1 for s in subs_all if speech_n[s] >= 4)} subjects with >= 4 speech")
    for k in (4, 6):
        cohort = [s for s in subs_all if speech_n[s] >= k]
        rows = []
        for name, allowed in FAMILY_SETS.items():

            def picker(s: str, a: tuple = allowed, kk: int = k) -> list | None:
                pool = fam_idx(s, a)
                if len(pool) < kk:
                    return None
                rng = np.random.default_rng(20260922 + (int.from_bytes(s.encode()[-6:], "little") % 100000))
                return sorted(rng.choice(pool, size=kk, replace=False).tolist())

            kept, halves, per = build(cohort, picker)
            rows.append((name, score(kept, halves), per))
        table(rows, f"B2. supply-matched at exactly {k} extents per subject (family mix is the only difference)")

    # B3 is the production question: one cohort, and each condition adds its families on top of
    # the speech the subject already has. Supply grows with the family set, so this is the only
    # comparison in which excluding a family is allowed to cost coverage.
    cohort = [s for s in subs_all if speech_n[s] >= 4]
    rows = []
    for name, allowed in FAMILY_SETS.items():
        kept, halves, per = build(cohort, lambda s, a=allowed: fam_idx(s, a))
        rows.append((name, score(kept, halves), per))
    table(rows, "B3. one cohort, additive supply: the production decision")


def main() -> None:
    """Run section A then section B over the exp_c per-extent vectors."""
    data = load(sys.argv[1])
    print(f"subjects loaded: {len(data)}")
    section_a(data)
    section_b(data)


if __name__ == "__main__":
    main()
