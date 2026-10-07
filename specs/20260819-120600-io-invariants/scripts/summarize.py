import json
from collections import OrderedDict

d = json.load(open("/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/matrix.json"))
R = d["results"]

ENC = ["torchcodec.AudioEncoder", "torchaudio.save", "soundfile.write", "ffmpeg-cli"]
DEC = ["torchcodec.AudioDecoder", "torchaudio.load", "soundfile.read(f32)", "soundfile.read(f64)", "librosa.load", "ffmpeg-cli"]
TGT = OrderedDict()
for r in R:
    TGT.setdefault(r["target"], None)


def get(t, e, s):
    for r in R:
        if r["target"] == t and r["encoder"] == e and r["signal"] == s:
            return r
    return None


def cell(c):
    if c is None:
        return "  -   "
    if "error" in c:
        return " ERR  "
    if c.get("exact"):
        return "EXACT "
    m = c.get("max_abs_diff", c.get("max_abs_diff_prefix"))
    if m is None:
        return " n/a  "
    if m == 0.0:
        return "  0e0 "
    return f"{m:6.1e}"


for sig in ["q16", "f32", "q16_stereo"]:
    print("=" * 130)
    print(f"### ROUND-TRIP: signal={sig}   (EXACT = bit-identical float32; else max |diff|; ERR = decoder failed)")
    print("=" * 130)
    hdr = f"{'target':<12} {'encoder':<24} " + " ".join(f"{x[:12]:>12}" for x in DEC)
    print(hdr)
    for t in TGT:
        for e in ENC:
            r = get(t, e, sig)
            if r is None:
                continue
            if r["write"] != "ok":
                print(f"{t:<12} {e:<24} WRITE FAILED")
                continue
            row = " ".join(f"{cell(r['decode'].get(dn)):>12}" for dn in DEC)
            print(f"{t:<12} {e:<24} {row}")
        print("-" * 130)

print()
print("=" * 130)
print("### OUT-OF-RANGE (signal peak 3.0, tail = flat 1.5).  peak = |max| read back; distinct = unique values; W = warnings at write")
print("=" * 130)
print(f"{'target':<12} {'encoder':<24} {'wr':<4} {'W':<3} " + " ".join(f"{x[:14]:>16}" for x in ["torchcodec.dec", "soundfile.read", "ffmpeg-cli"]))
for t in TGT:
    for e in ENC:
        r = get(t, e, "oor")
        if r is None:
            continue
        if r["write"] != "ok":
            print(f"{t:<12} {e:<24} FAIL")
            continue
        nw = len(r.get("write_warnings", []))
        parts = []
        for dn in ["torchcodec.AudioDecoder", "soundfile.read(f32)", "ffmpeg-cli"]:
            c = r["decode"].get(dn, {})
            if "error" in c:
                parts.append(f"{'ERR':>16}")
            else:
                parts.append(f"{'pk=' + format(c.get('got_peak') or 0, '.4f') + ' n=' + str(c.get('got_distinct')):>16}")
        print(f"{t:<12} {e:<24} {'ok':<4} {nw:<3} " + " ".join(parts))
    print("-" * 130)

print()
print("### SAMPLE RATE / CHANNEL preservation (signal=q16_stereo unless noted)")
for t in TGT:
    for e in ENC:
        r = get(t, e, "q16_stereo")
        if r is None or r["write"] != "ok":
            continue
        c = r["decode"].get("torchcodec.AudioDecoder", {})
        sfi = r.get("sf_info", {})
        print(
            f"{t:<12} {e:<24} sr_written={r.get('probe',{}).get('sample_rate')} "
            f"sr_libsndfile={sfi.get('sr')} ch={r.get('probe',{}).get('channels')} "
            f"tc_read_sr={c.get('sr_out')} shape={c.get('got_shape')} ref={c.get('ref_shape')}"
        )

print()
print("### decoder ERRORS")
seen = set()
for r in R:
    if r["write"] != "ok":
        continue
    for dn, c in r.get("decode", {}).items():
        if "error" in c:
            k = (r["target"], r["encoder"], dn, c["error"][:80])
            if k[:3] in seen:
                continue
            seen.add(k[:3])
            print(f"{r['target']:<12} {r['encoder']:<24} {dn:<24} {c['error'][:180]}")

print()
print("### any write-time warnings anywhere?")
any_w = [(r["target"], r["encoder"], r["signal"], r["write_warnings"]) for r in R if r.get("write_warnings")]
print(any_w if any_w else "NONE — no encoder emitted a Python warning in any cell, including out-of-range")

print()
print("### decode-time warnings")
dw = set()
for r in R:
    for dn, c in r.get("decode", {}).items():
        for w in c.get("warnings", []):
            dw.add((dn, w[:120]))
for x in sorted(dw):
    print(x)
