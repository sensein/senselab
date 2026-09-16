# The picker's real failure mode is a frozen index, not a stranded one

Investigation only — `cuda_probe.py` and every other source file are unchanged. This spec answers
"what would a routing change do" so that question can be decided before any code moves.

## The trigger

An ORCD L40S node under `mit_preemptable` reports driver 590.48.01, CUDA 13.1. `_PYTORCH_INDEX_MAP`
(`src/senselab/utils/cuda_probe.py:32-41`) tops out at `cu128` (12.8), so `pick_torch_index`
(`cuda_probe.py:153-216`) falls through its whole table and returns `cu128` — the newest entry
`<=` the host's version (`cuda_probe.py:203-204`). The question is whether `cu128` is actually the
right choice for that host, and whether adding `cu129`/`cu130` entries would help.

## Method

Every subprocess-venv call site under `src/senselab/` was found via `grep -rn "ensure_venv("` and
its `torch`/`torchaudio` spec and any `max_cuda_version` read from source (line numbers below).
Two venv pairs collapse to one row each because they share a venv name and an identical spec:
`features_extraction/sparc.py` + `voice_cloning/sparc.py` both build `"sparc"`; `voice_cloning/coqui.py`
+ `text_to_speech/coqui.py` both build `"coqui"`; `speech_to_text/nemo.py` + `speaker_diarization/nvidia.py`
both build `"nemo-diarization"`. `classification/yamnet.py` and `health_acoustics/hear.py` were
checked and excluded — both pin `tensorflow`, not `torch` (`yamnet.py:48-51`, `hear.py:179`), so
`_torch_install_specs` (`subprocess_venv.py:664-683`) finds no torch spec there and `ensure_venv`
never probes CUDA for them at all (`subprocess_venv.py:263-273,295-298`). 19 torch-bearing rows
remain.

For each of `cpu`/`cu118`/`cu121`/`cu124`/`cu126`/`cu128`/`cu129`/`cu130`, `torch`'s and
`torchaudio`'s full `/whl/<index>/<pkg>/` listing was fetched over plain HTTPS (`curl`, no
authentication, no install) and parsed for `linux_x86_64` wheel filenames — the platform tag on the
brief's L40S node and on every GPU CI/cluster target senselab runs on. Each backend's spec string
was evaluated against the parsed version set with `packaging.specifiers.SpecifierSet`, taking the
highest satisfying version per `(index, package)` pair — the same "newest satisfying" behavior `uv
pip install` exercises during `ensure_venv`'s Stage 1 (`subprocess_venv.py:412-427`). Script and raw
index pages: `/private/tmp/claude-501/.../scratchpad/cuda_matrix_*` (`cuda_matrix_build.py`,
`cuda_matrix_<index>_<pkg>.html`) — not committed, reproducible from the URLs above.

## Correcting the brief's inputs before relying on them

The brief's torch-only version table is right — verified independently, matches to the patch
version on all 8 indexes including `cu118`, which the brief didn't list. But it omits the fact that
matters most: **`torchaudio`'s newest version is `2.11.0` on every index that carries
it at all — `cpu` included.** Torch kept shipping (`2.12.0`, `2.12.1`, `2.13.0`, `2.14.0`) after
`torchaudio` stopped. This isn't a bug; PyTorch's own tracker says so:

> "TorchAudio 2.11 marks the finalization of the TorchAudio migration and is compatible with torch
> 2.11 and future versions of torch." — [pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)

`torchaudio` moved into "maintenance phase," decode/encode moved to `torchcodec` (features
deprecated in 2.8, removed in 2.9 — same issue), and rather than bumping its own version to track
torch's, the project re-publishes the **same** `2.11.0` under each new CUDA tag: `torchaudio-2.11.0+cu126`,
`+cu128`, `+cu129`, `+cu130` all exist (confirmed by listing). So on any index whose torch resolves
above `2.11.0`, an open-ended `torchaudio` spec resolves to `2.11.0` while torch resolves higher —
a real version-number split, but *not* the CUDA-toolchain ABI break `cuda_probe.py` exists to
prevent, because both wheels still come from Stage 1's single `--index-url` call
(`subprocess_venv.py:397-427`) and both carry the *same* `+cuXXX` local tag. The mismatch this
module actually guards against — `torch==X+cu129` against a tagless PyPI `torchaudio==X`
(`qwen.py:39-41`, `subprocess_venv.py:386-395`) — doesn't reoccur here as long as Stage 1 keeps
naming one index for both packages, which it does regardless of which index gets picked. The
brief's request to "report any backend where torch and torchaudio resolve to versions that don't
pair" is answered below, but the pairing that matters (same `+cuXXX` suffix) always holds; only the
cosmetic version number diverges.

## The matrix

Columns are the brief's requested set. Cell format `torch / torchaudio`, both the highest version
satisfying that backend's spec on that index; `no wheel` = spec unsatisfiable on that index.
`child-adult-diarization` and `unasdiff` pin torch with `==`, which is closed despite having no
`<`; corrected in the "open?" column below (my scratch script's `"<" not in spec` heuristic
mislabeled both as open — a script bug, not a data error, fixed by hand here).

| backend | torch spec | open? | `max_cuda_version` cap | cpu | cu121 | cu124 | cu126 | cu128 | cu129 | cu130 |
|---|---|---|---|---|---|---|---|---|---|---|
| clearvoice¹ | `>=2.0.1` | open | — | 2.14.0 / 2.11.0 | 2.5.1 / 2.5.1 | 2.6.0 / 2.6.0 | 2.14.0 / 2.11.0 | 2.11.0 / 2.11.0 | 2.13.0 / 2.11.0 | 2.14.0 / 2.11.0 |
| clearvoice-speechscore | `>=2.0.1` | open | — | 2.14.0 / 2.11.0 | 2.5.1 / 2.5.1 | 2.6.0 / 2.6.0 | 2.14.0 / 2.11.0 | 2.11.0 / 2.11.0 | 2.13.0 / 2.11.0 | 2.14.0 / 2.11.0 |
| continuous-ser | `>=2.8` | open | — | 2.14.0 / n/a² | no wheel | no wheel | 2.14.0 / n/a² | 2.11.0 / n/a² | 2.13.0 / n/a² | 2.14.0 / n/a² |
| ppgs | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| sparc³ | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| brouhaha | `>=2.0,<2.3` | closed | (12,1) | 2.2.2 / 2.2.2 | 2.2.2 / 2.2.2 | no wheel | no wheel | no wheel | no wheel | no wheel |
| unasdiff | `==2.6.0` | closed | (12,4) | 2.6.0 / 2.6.0 | no wheel | 2.6.0 / 2.6.0 | 2.6.0 / 2.6.0 | no wheel | no wheel | no wheel |
| child-adult-diarization | `==2.3.0` | closed | (12,1) | 2.3.0 / 2.3.0 | 2.3.0 / 2.3.0 | no wheel | no wheel | no wheel | no wheel | no wheel |
| diarizen | `>=2.1,<2.9` | closed | — | 2.8.0 / 2.8.0 | 2.5.1 / 2.5.1 | 2.6.0 / 2.6.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| moss-transcribe-diarize | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| nemo-diarization⁴ | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| driftse | `>=2.3` | open | — | 2.14.0 / 2.11.0 | 2.5.1 / 2.5.1 | 2.6.0 / 2.6.0 | 2.14.0 / 2.11.0 | 2.11.0 / 2.11.0 | 2.13.0 / 2.11.0 | 2.14.0 / 2.11.0 |
| nemo-canary-qwen | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| crisperwhisper | `>=2.4` | open | — | 2.14.0 / 2.11.0 | 2.5.1 / 2.5.1 | 2.6.0 / 2.6.0 | 2.14.0 / 2.11.0 | 2.11.0 / 2.11.0 | 2.13.0 / 2.11.0 | 2.14.0 / 2.11.0 |
| qwen-asr | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| s3prl | `>=2.0,<2.5` | closed | — | 2.4.1 / 2.4.1 | 2.4.1 / 2.4.1 | 2.4.1 / 2.4.1 | **no wheel** | **no wheel** | **no wheel** | **no wheel** |
| coqui⁵ | `>=2.8,<2.9` | closed | — | 2.8.0 / 2.8.0 | no wheel | no wheel | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | 2.8.0 / 2.8.0 | **no wheel** |
| qwen-tts | `>=2.6` | open | — | 2.14.0 / 2.11.0 | no wheel | 2.6.0 / 2.6.0 | 2.14.0 / 2.11.0 | 2.11.0 / 2.11.0 | 2.13.0 / 2.11.0 | 2.14.0 / 2.11.0 |
| pii-detection | `>=2.8,<2.9` | closed | — | 2.8.0 / n/a² | no wheel | no wheel | 2.8.0 / n/a² | 2.8.0 / n/a² | 2.8.0 / n/a² | **no wheel** |

¹ `utils/clearvoice.py:70-75`. ² `continuous-ser` and `pii-detection` name no `torchaudio` spec
(`speech_emotion_recognition/api.py:142-149`; `pii_detection/subprocess_backend.py:90-95` says so
explicitly — "neither Presidio nor GLiNER decode audio") — `_torch_install_specs` forwards only
what's named (`subprocess_venv.py:664-683`), so Stage 1 installs `torch` alone for both; the SER
worker imports `soundfile`, not `torchaudio` (`api.py:159`). Intentional in both cases, not a gap.
³ `features_extraction/sparc.py:31-35`
+ `voice_cloning/sparc.py:27-31`, identical spec, same venv name `"sparc"`. ⁴ `speech_to_text/nemo.py:23-33`
+ `speaker_diarization/nvidia.py:23-27`, same venv name `"nemo-diarization"`. ⁵ `voice_cloning/coqui.py:26-30`
+ `text_to_speech/coqui.py:28-32`, same venv name `"coqui"`.

Other citations: `continuous-ser` `speech_emotion_recognition/api.py:142,146`; `ppgs`
`features_extraction/ppg.py:83,86-87`; `brouhaha` `scene_quality/brouhaha.py:62,65-66,82`;
`unasdiff` `source_separation/unasdiff.py:131,140,155-156`; `child-adult-diarization`
`speaker_diarization/child_adult.py:72,75-76,89`; `diarizen` `speaker_diarization/diarizen.py:81,85-86`;
`moss-transcribe-diarize` `speaker_diarization/moss.py:50,53-54`; `driftse`
`speech_enhancement/driftse.py:46,54-55`; `nemo-canary-qwen` `speech_to_text/canary_qwen.py:37,52-53`;
`crisperwhisper` `speech_to_text/crisperwhisper.py:34,51,53-64` (both platform branches pin the same
`torch>=2.4`/`torchaudio>=2.4`); `qwen-asr` `speech_to_text/qwen.py:29,42-43`; `s3prl`
`ssl_embeddings/s3prl.py:23,26-27`; `qwen-tts` `text_to_speech/qwen_tts.py:142,153-154`;
`pii-detection` `text/tasks/pii_detection/subprocess_backend.py:77,90-95,110` (no `torchaudio` spec —
deliberately excluded, see footnote 2).

## The hypothesis: confirmed for `cu128`, reversed for `cu130`

**Confirmed, for today's table.** Every open-ended backend (`clearvoice`, `clearvoice-speechscore`,
`continuous-ser`, `driftse`, `crisperwhisper`, `qwen-tts`) diverges between `cpu` and `cu128`
(`cu128`'s torch caps at `2.11.0`; `cpu`'s reaches `2.14.0`). Every closed-band backend without a
cap matches exactly between `cpu` and `cu128` (`2.8.0`/`2.8.0` on both) — this is `qwen-asr`'s
situation, shared by seven more `torch>=2.8,<2.9` backends and by `diarizen`'s `>=2.1,<2.9`,
matching the brief's `qwen-asr` example.

**Reverses under a table extended with `cu130`.** `cu130`'s torch reaches `2.14.0` — parity with
`cpu` — so *every open-ended backend's* cpu/device divergence disappears if the host is routed to
`cu130` instead of `cu128`. But `cu130` carries no `2.8.x` torch build at all (PyPI never published
one — `cu130` didn't exist when `2.8.0` was current), so **all eight `torch>=2.8,<2.9`-band
backends, plus `diarizen` (`>=2.1,<2.9`) — nine rows — go from an exact `cpu` match today to `no
wheel` at all** if a CUDA-13.1 host is routed there. The openness of a spec does not reliably predict
divergence in general; it predicts divergence against `cu128` specifically, because `cu128` is the
one member of the table that stopped receiving new torch builds after `2.11.0` while its neighbors
kept moving — see below.

## `cu128` isn't merely "stranded" between two live neighbors — it's frozen

The brief frames `cu128` as bracketed by an older index (`cu126`) and newer ones (`cu129`/`cu130`)
that both carry `2.14.0`/`2.13.0`. That's true, but it understates the mechanism: **`cu128` itself
stopped getting new torch releases after `2.11.0`**, while `cu126` — an *older*, already-existing
index — kept receiving every subsequent release through `2.14.0`. This is not a version-window
gap that a wider table papers over; the same thing could recur for `cu129` or `cu130` once *they*
stop being upstream's actively-built index. `pick_torch_index`'s "highest index `<=` host" rule
(`cuda_probe.py:203-210`) has no way to know an index went stale — it only compares version
numbers — so any static table will eventually mis-route a host whose CUDA sits at or above a
frozen index's tag, for as long as that index remains the nearest `<=` entry. `cu128` is simply the
first instance senselab has hit.

## Already broken today, independent of any table change

**`s3prl`** (`torch>=2.0,<2.5`, `ssl_embeddings/s3prl.py:26-27`) declares no `max_cuda_version`.
Its highest-satisfying torch is `2.4.1`, available only on `cpu`/`cu118`/`cu121`/`cu124` — **not**
`cu126`, `cu128`, `cu129`, or `cu130`. `_PYTORCH_INDEX_MAP` already contains `cu126` and `cu128`
today (`cuda_probe.py:34-35`), so **any host reporting CUDA `>= 12.6` already gets `no wheel` for
`s3prl`**, right now, with zero table changes — the same class of defect `brouhaha` (`(12,1)` cap),
`unasdiff` (`(12,4)` cap), and `child-adult-diarization` (`(12,1)` cap) were already given caps for.
`s3prl` is the one backend in this set that needs one and doesn't have it. This is independent of
the CUDA-13.1 / `cu129`/`cu130` question the brief raises and should be fixed regardless of what
happens to the table's upper end.

**The nine `<2.9`-closed-band backends are also exposed on the *low* end today**, not just the high
end the brief asks about: on any host whose selected index is `cu121` or `cu124` (CUDA in roughly
`[12.1, 12.6)` — an older L4/A10-class driver stack, not hypothetical), `torch>=2.8,<2.9` already
has `no wheel` (see the matrix: `ppgs`/`sparc`/`coqui`/`qwen-asr`/etc. all read `no wheel` under
`cu121` and `cu124`). `max_cuda_version` only *caps* a selection downward; there is no symmetric
"floor" concept in `pick_torch_index`, so a backend whose torch floor requires a newer index than
the host offers has no escape hatch today. This is a structural gap, not a `cu129`/`cu130`
regression, and is worth its own follow-up — flagged here, not resolved.

## Recommendation

**1. No single new top-of-table index suits every backend.** Routing a CUDA-13.1 host to `cu130`
(closes the open-ended backends' `cpu` gap) breaks nine closed-band rows outright; leaving it
at `cu128` (today's behavior) keeps those nine intact but leaves the six open-ended rows capped
below `cpu` parity for no wheel-availability reason. Between the two entries the brief names,
neither is uniformly right — this is the "messy answer" the brief anticipated. A third option
this investigation surfaced and the brief didn't ask about: **`cu129` matches `cu126`/`cu128`
exactly for every closed-band row** (all read `2.8.0`/`2.8.0`, none read `no wheel`) while
narrowing (not closing) the open-ended gap to `2.13.0` vs `cpu`'s `2.14.0`. If the table is
extended at all, `cu129` is the strictly-safer next entry; `cu130` should not be added without
first giving every closed-`<2.9`-band backend (and `diarizen`) an explicit `max_cuda_version` cap,
exactly mirroring what `brouhaha`/`unasdiff`/`child-adult-diarization` already do for their older
pins (`cuda_probe.py:172-182` already documents this exact pattern for `brouhaha`).

**2. `cu126` over `cu130`, on current evidence.** Restricting to the 16 backends with no
`max_cuda_version` cap (the 3 capped rows never route through `cu126` regardless of table
extension, so comparing their `cu126` cell to `cpu` isn't the relevant test), `cu126` reproduces
`cpu`'s resolved version *exactly* for 15 of 16 (the sole exception, `s3prl`, is already broken on
`cu126` today for reasons unrelated to this question — see above). `cu130` is a CUDA
**major**-version jump that none of these 19
venvs has ever been built or run against — it has no track record here, only a wheel listing.
Evidence that would justify moving the default to `cu130` rather than assuming it works: (a) an
actual run of a representative open-ended backend (e.g. `driftse`, the simplest of the six) and a
representative closed-band backend (once capped) against a `cu130`-installed torch+torchaudio pair
on a real `cu130`-class host — not just a successful `uv pip install`; (b) confirmation that the
measured host, driver 590.48.01 / CUDA 13.1, is within NVIDIA's and PyTorch's supported combination
for `cu130` wheels, since a CUDA major bump is exactly the kind of change where driver/runtime
surprises appear that a minor-version bump (`cu126`→`cu128`→`cu129`) hasn't shown so far; (c) a
second `cu130` host measurement some weeks apart, so a frozen-index repeat of the `cu128` situation
found here isn't mistaken for a validated index the first time it's checked.

**3. The trade-off in the closed `<2.9` bands, stated but not resolved.** Pinning `<2.9` makes nine
backends' resolved torch identical across `cpu`/`cu126`/`cu128`/`cu129` today — no device
divergence, no surprise. The cost, sharpened by this investigation: it also means those nine
backends get **zero forward path** past torch `2.8` on any index — including `cu126`, which already
carries `2.14.0` and would serve every one of them a newer, presumably-supported release if the
ceiling were lifted — and it means the same nine backends are the ones that go from "fine" to
"no wheel at all" the moment the table's preferred index moves past whichever index still carries
`2.8.x` (currently `cu129`; not `cu130`). A closed band buys today's stability at the cost of a
brittleness that only shows up the next time the table changes. Whether that trade is worth it is
the owner's call, not this spec's.

**4. Backends that would break under an extended table** (i.e., the load-bearing question,
mirroring `cuda_probe.py:176-182`'s existing `brouhaha` precedent): if `cu130` becomes the
preferred index for a CUDA-13.1-class host, these nine currently-uncapped rows get `no wheel` and
raise `SenselabCudaCompatibilityError` (`cuda_probe.py:71-104`) — `ppgs`, `sparc`,
`moss-transcribe-diarize`, `nemo-diarization`, `nemo-canary-qwen`, `qwen-asr`, `coqui`,
`pii-detection` (all `torch>=2.8,<2.9`), and `diarizen` (`torch>=2.1,<2.9`). Each would need a
`max_cuda_version` of at least `(12, 9)` (routes to `cu129`, confirmed to carry `2.8.0`) to survive
a `cu130`-preferring table, the same fix already applied to `brouhaha` (`(12,1)`), `unasdiff`
(`(12,4)`), and `child-adult-diarization` (`(12,1)`) for their older pins. If `cu129` is chosen as
the new top entry instead of `cu130`, none of these nine needs a new cap — confirmed directly in
the matrix (their `cu129` column already reads `2.8.0`/`2.8.0`, not `no wheel`).

## Rejected framing

Treating the `torch`/`torchaudio` version-number split (`2.14.0` vs `2.11.0`) on open-ended
backends as an ABI defect, per the brief's framing, was the first read of the data and is wrong:
`subprocess_venv.py`'s Stage 1 installs both packages from one named index in one command
(`subprocess_venv.py:412-427`), so they always share the same `+cuXXX` local tag regardless of
which numeric version each resolves to, and torchaudio's own finalization notice
([pytorch/audio#3902](https://github.com/pytorch/audio/issues/3902)) states explicit forward
compatibility with newer torch. The real, still-open risk in that neighborhood is the one
`subprocess_venv.py:386-395` already documents and `qwen.py:39-41` already saw in production
(PyPI's *unpinned* transitive resolution splitting `torch==X+cu129` against a tagless
`torchaudio==X`) — a Stage-2/transitive problem the constraint file at
`subprocess_venv.py:478-481` (added after the `unasdiff` H100 incident,
`subprocess_venv.py:459-469`) already guards against, not a Stage-1 index-selection problem.
