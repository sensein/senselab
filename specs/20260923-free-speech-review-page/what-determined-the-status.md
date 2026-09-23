# What determined the status

A reader looking at a withheld card could see *that* it was withheld and nothing about *why*. This
is what the page now shows, and the corpus-wide shape of it.

## The worked case

`sub-004d42e9-…`, `cinderella-story`. The owner asked: *"if both gates and LLM flagged this we need
to know why"*. Read from the store, the answer is that **neither did**.

| | |
| --- | --- |
| triage | `pass` |
| release | `withheld` |
| `release_ground` | `null` |
| REDACT's own verdict | **`fail`** — *verification found pii on the redacted transcript: `PERSON`* |
| gates evaluated | 2 — `response_min_s` read 242.085 s against a bound of 0.5 (**passed**); `dominant_speaker_share_min` read 0.936 against 0.9 (**passed**) |
| gates declared, never evaluated | 4 — `coverage_min`, `echo_overlap_max`, `gap_off_task_min_s`, `verbatim_overlap_max` |
| LLM reviewer | **never ran** — `status: not_run`, `iterations: 0`, `model_id: ""` |
| live findings | 19, resolving to 12 marks |
| categories | `NAME` 9, `PERSON` 7, `MISC` 2, `DATE_TIME` 1 |
| detectors | `presidio` 8, `rules/ner` 5, `rules/gazetteer+ner` 4, `rules/ner+rareword` 2 |
| haystack | 17 on the consensus transcript, 2 on `asr_qwen` |
| `in_stimulus` | **`null` on all 19** |
| exemptions | `expected_speech_declared: false`, 0 of 19 exempted |

So the chain is: the detectors flagged 19 things, none of which could be checked against a stimulus
because `cinderella-story` declares none; REDACT redacted them, re-scanned the redacted transcript,
still saw `PERSON`, and returned `fail`; the fold turned that into `withheld`. Every gate that was
evaluated passed. The reviewer never ran — and could not have, because the detector path had
already withheld, which is exactly what its `failure` string says. Nothing corroborated the
detectors.

A note on the two counts: 19 is the number of live `pii` entities, 12 the number of marks the page
shows, because a mark merges adjacent words of one category and because the same finding can be
reached on both the consensus and a per-recognizer haystack. Both are right; they count different
things, and the panel says 19 because that is what `redaction_exemptions.n_findings` also says.

## Why `release_ground` is blank exactly where it matters

`_release_from` sets `release_ground` only when the *fold* decides. When REDACT wrote a verdict, its
outcome maps straight through — `pass` → `releasable`, `flag`/`fail` → `withheld` — and the ground
is `None` by construction. So a page showing `release_ground` alone is blank on every releasable
and every withheld recording, which is precisely the set a reader cares about. The account has to
fall back to REDACT's own `why`, and the panel says why the ground is empty rather than leaving an
absence to be misread as missing data.

## Three distinctions the page has to make

**A gate that failed, one that could not be answered, and one never evaluated.** `passed` is `True`,
`False`, or the string `UNDETERMINED` — applied, and unanswerable because nothing measured the
reading or nobody has measured the bound. A gate the config names for the group but that never
appears in `applied` was never evaluated at all. None of the last three is a failure, and the panel
says which is which. **This caught a bug in the first version of the panel**: `UNDETERMINED` is a
truthy string, so a boolean coercion rendered an unanswerable gate as a passing one. 292 gates in
the corpus are `UNDETERMINED`.

**A reviewer that ran and found nothing, one that never ran, and one that could not load.** Five
statuses: `clean` is the only one meaning "it read the transcript and flagged nothing"; `disabled`
and `not_run` mean it never ran; `absent` means it tried and the model would not load; `flagged`
means it found something. The panel gives `disabled`/`not_run`/`absent` a highlighted box, because
silence from a reviewer that never ran carries no information and must not be read as agreement.

**A finding checked against the stimulus, one checked and absent from it, and one that could not be
checked.** `in_stimulus` is `True`, `False` or `None`. The first version of the extract coerced it
to `bool`, which collapsed `None` into `False` and hid the whole third artefact family — see
`no-stimulus-to-check-against.md`.

## What the corpus looks like through this lens

| | |
| --- | --- |
| release × REDACT outcome | `releasable`/`pass` 3,719 · `withheld`/`fail` 2,972 · `nothing_to_redact`/no verdict 5,009 · `not_assessed`/no verdict 1 |
| LLM reviewer | `disabled` 3,719 · `not_run` 2,972 · no annotation 5,010 · **`clean` 0 · `flagged` 0 · `absent` 0** |
| evaluated gates | 21,492 passed · 1,607 failed · 292 unanswerable |
| gates declared but never evaluated | 4 per recording on 11,695 of 11,701 |
| withheld recordings with no failing gate | **2,648 of 2,972 (89.1%)** |

Two things worth saying out loud.

**The LLM reviewer ran on zero recordings in this corpus.** Not once. Where REDACT passed it was
`disabled` by configuration; where REDACT failed it was `not_run` because there was nothing left to
release. So the owner's question generalises past the one recording: across all 11,701, nothing
corroborated the detectors, and a page that left the reviewer's state implicit would let that pass
unnoticed.

**Withholding is essentially independent of the gates.** 89.1% of withheld recordings have no
failing gate at all; the decision is REDACT's post-redaction verification, and the release axis
never touches triage. The release and triage axes are genuinely separate, and the page now shows
both rather than one word that reads as if it summarised the other.

## How it is delivered

The account is per recording and would cost more than the transcripts if written into every card.
It is interned instead: nine node names, a few dozen `why` sentences, one gate specification per
group and one `ran` map are pooled once and referenced by index; only a gate's measured value and
its outcome vary per recording. The panel is built on demand from that payload when the reader asks
for it, so the DOM stays the size it was.
