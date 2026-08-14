# PII detection

Scan a `str`, a `ScriptLine`, or a sequence of either for personally identifiable
information.

## Scanning and deciding are separate calls

`scan_for_pii` **runs the detectors** and returns a `PiiScan`: the spans they found, which
detectors actually ran, and what failed. It reaches no conclusion — there is no
`contains_pii` on a `PiiScan`, and no confidence.

`decide_pii` **aggregates that evidence into a verdict**: how many detectors must agree
before a finding counts, how per-detector scores and agreement combine into one confidence,
and what `contains_pii` becomes.

`detect_pii` is the composition of the two, for callers who want the default decision.

The split is deliberate, and the parameter lists are the reason. A Presidio score floor or a
GLiNER label set describes *how to run a tool*; a corroboration requirement is *a judgement
about what its output means*. Mixing them in one signature makes the second look like the
first — a knob among knobs — when it is the part a downstream consumer is most likely to
need to change. Keeping them apart means a caller can take the evidence and apply their own
rule (a severity ordering this module declines to impose, an aggregation across several
transcripts of one recording) without re-running detection or arguing with a verdict that
was already computed:

```python
scan = scan_for_pii(transcript)                       # execution
report = decide_pii(scan, require_cross_source_corroboration=False)   # decision
```

`detect_pii_in_audios` in `senselab.audio.tasks.pii_detection` is the transcribe-then-detect
composition on top of these.

Nothing here knows about a run "pass", a per-pass tag, or an ASR ensemble. A caller that
needs to merge several ASR backends' transcripts for one recording and corroborate across
them builds an adapter on top — `audio_analysis` keeps its own for exactly that reason.

## Four detectors, and why each is here

Three run inside an isolated subprocess venv; the fourth runs in the host process and is
off by default.

**Microsoft Presidio Analyzer** — regex plus a spaCy NER orchestrator, with purpose-built
recognizers for emails, phone numbers, SSNs, credit cards, IP addresses, dates and
locations. Strong on structured identifiers, weak on the PHI categories that have no
regular form.

**GLiNER PII** (`nvidia/gliner-pii`) — a zero-shot transformer NER fine-tuned on ~100k
synthetic PII/PHI records, defaulting to the HIPAA Safe Harbor 18. It catches what
Presidio has no recognizer for: medical record number, health plan number, account
number, fax number, biometric / device / vehicle identifiers.

**Rules cascade** (`rules.py`, ported from PR #542) — regex, gazetteers, spaCy NER,
self-disclosed demographics, age-over-90, and a combinatorial re-identification window.
It earns its place next to two model-based detectors through its *precision guards*
rather than its recall: a Zipf-frequency hard gate on common-word names, structured
identifier format validation, and holiday reclassification. A false positive is what
makes a PII tool unusable — more than a false negative is.

**Optional local LLM** (`local_llm.py`) — off by default and loopback-only, enforced at
construction rather than documented. It is `"llm"` in `detectors`; `default_detectors()`
omits it because a default-on network detector would make a scan depend on whether a
server happened to be listening, and the same corpus would score differently on two
machines with nothing in the report explaining the gap. An unreachable server is a
recorded failure, never a clean pass.

## Why a subprocess venv

`presidio-analyzer`, `spacy` and `gliner` pull a dependency set that does not resolve on
every Python the host may be running, and spaCy in particular has no wheel for some of
them. Detection therefore runs in an isolated Python 3.13 venv built on first use, and
the host process imports none of it. `subprocess_backend.py` owns that boundary.

The rules cascade's source travels *with* the request rather than being duplicated inside
the worker script — two copies of a 900-line cascade would drift the moment either was
edited. It is sent whenever the rules **or** GLiNER detector is on, because GLiNER needs
`_gliner_chunks` from it.

## Long transcripts are windowed

GLiNER checkpoints cap input at a few hundred subword tokens and silently truncate past
it — a long transcript scanned whole loses its tail with no error, which reads downstream
as "no PII down there". `_gliner_chunks` splits the text into overlapping word windows,
each carrying its absolute character offset so a span can be re-based onto the original;
a window-relative offset would make the masked preview redact the wrong characters. The
overlap exists so an entity sitting on a boundary appears whole in at least one window.
Spans are deduped across windows keeping the highest score, since a window containing an
entity whole should outrank one that caught its edge.

## The GLiNER label list must stay flat

**Do not add overlapping labels.** `nvidia/gliner-pii` exhibits competing-claim
interference: when two labels can plausibly cover the same span, the model commits the
span to one and silently drops it from the other — *even when the other would have been
correct*. Measured on `john.doe@example.com`:

| Labels passed | Result |
| --- | --- |
| `[person, first_name, last_name, email, email_address, …]` | `John` / `Doe` as first/last_name at 1.0; the full `john.doe@example.com` got **no** email span at any score above 0.0 |
| `[name, address, date, phone_number, email, ssn, …]` (flat, one `name`, one `email`) | `John` / `Doe` as `name` at 1.0 **and** `john.doe@example.com` as `email` at 1.0 |

Same model, same threshold, same input. The only difference was label overlap. Growing
the list past the HIPAA-18 is how you lose detections you already had.

## Confidence, and its denominator

`detection_confidence` is a continuous `[0, 1]` value combining three signals per unique
`(category, normalized_text)` finding: the maximum raw detector score on it, cross-detector
agreement, and cross-ASR-model agreement where the caller supplied more than one transcript.

The agreement denominator is **the number of detectors that actually ran for this report**,
not the number that exist. Using the known-detector count would silently deflate every
confidence the moment a fourth detector was added — which is why `_KNOWN_DETECTORS` growing
to include `"llm"` does not move any existing score.

`None` and `0.0` are different answers: `None` means the detectors did not run (subprocess
failure, `detectors=[]`, or every detector failing to load), `0.0` means they ran and found
nothing. A caller that collapses the two turns "we never checked" into "we checked and it
was clean".

## No category-severity weighting, deliberately

It is tempting to weight `US_SSN` and `CREDIT_CARD` above `PERSON`. In pediatric and
clinical voice data those categories have a near-zero true-positive rate and are dominated
by ASR digit hallucinations, so weighting them up would inflate exactly the hits a reviewer
should de-prioritize. Severity ordering is the caller's to apply with knowledge of their
corpus.

## Corroboration

With `require_cross_source_corroboration=True` (the default) and ≥2 detectors running,
`contains_pii` flips only for a `(category, normalized_text)` pair that ≥2 detectors
independently flagged. Agreement keys on a coarse *family* rather than the raw category, so
two detectors naming the same entity slightly differently still corroborate. With one
detector running, corroboration cannot apply and any single hit counts — one witness is not
a quorum, but it is also not nothing.
