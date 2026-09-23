# The file-level verdict

The last fold, and **the graph's only decision about the recording**. Reads the
[element store](store.md) — the deciding nodes' verdicts, the reporting nodes' `branch_report`
entities, the spans those nodes proposed, the branch decisions [`routing.md`](routing.md) wrote, and
the `ruleset_routing` measurement behind them — and answers what the graph concluded.

## A branch reports; this fold decides — owner decision, 2026-09-16

> "branches should just return branch spans, task conformance/deviation, and then downstream verdict
> can aggregate across branches (including quality) for decisions."
> "a refusal is a decision. a branch does not decide. it's an authority on task/branch specific
> detection."

Until 2026-09-16 every branch returned an `Outcome` — `pass` / `flag` / `fail` — and this fold then
decided *about that decision*: `_agreement` scored each branch outcome against its route, and
`_resolved` mapped every non-`FAIL` outcome to `present`. That double layer is gone.

A branch now returns exactly three things, and `QUALITY` returns them too:

| what | where it goes |
| --- | --- |
| **branch spans** | proposed into the branch's own family, in the store. `findings` is read off them |
| **task conformance** | `true` \| `false` \| `UNDETERMINED`, on the `branch_report` entity |
| **deviations** | typed and located, on the same entity. Recorded; never a flag ground |

A branch writes **no `Outcome`** and **no `why`**. It also **never raises over an unmeasured
operating point**: it names the point in the report's `unmeasured` and leaves the dependent
conformance `UNDETERMINED`, because a refusal is a decision. A misspelled key still raises
(`UnknownConfigKey`), as does an absent store entity the code assumed — neither is a detection.

`Outcome` survives for the nodes that *do* decide: ADMIT, PREPROCESS, TAXONOMY, `routing`, REDACT.

## What each input contributes

| input | what it contributes | flags? |
| --- | --- | --- |
| conformance `false` | **this fold's own reading**, from the branch's measurements against the gates for that task — not a claim the branch makes | yes, gated on the referent and the declared family |
| conformance `true` | nothing | no |
| conformance `UNDETERMINED` | nothing | no, unless `verdict.undetermined_flags` |
| the proposed spans | `findings`: `present` with one or more, `absent` where the branch reported and proposed none, `uncertain` where it left no report | only through a route mismatch |
| the route | `agreement`, against the spans | a `mismatch` flags |
| deviations | recorded in `deviations`, per node | **no** — see the ground-truth rule below |
| `unmeasured` | recorded per node | yes, under `verdict.unmeasured_points_flag` |
| REDACT's LLM re-read | `llm_redaction`, the annotation whole | `flagged` only, under `verdict.llm_redaction_flags`, and only on the **triage** axis |
| the declared task family | the key every conformance ground is read against | — |

**Deviations are recorded and are not folded into the flag column until ground truth exists.**
`specs/20260913-branch-contract-and-hints/design.md` states that `filler` and `stimulus_mismatch` are
expected on ordinary read speech, so routing them in would flag the corpus — and carries, since
2026-09-16, which of the ten declared types that argument actually fits and which are genuine
departures awaiting a measured rate. The constraint survived
this change and is now a declared switch — `verdict.deviation_flags`, shipping `false` — rather than
an implicit rule in the code, so that flipping it is a visible decision with a derivation.

**Every threshold that turns a reading into a judgement is in the `verdict:` config section**, not in
`branch:`. Three keys moved there on 2026-09-16 for that reason: `min_contrast_db`,
`tilt_max_db_per_octave` and `level_min_dbfs`. Owner-directed 2026-09-21, the remaining sixteen
follow, and they arrive keyed by task rather than flat — see below.
`config-derivations.md` § verdict carries every value's derivation.

## The gates, keyed by task — owner decision, 2026-09-21

*All gates should be in verdict not in branches and it should be task group related. Branches should
just provide the info necessary for the gates. So instruments sit in preprocess and branches.*

A **gate** says what reading is good enough and lives here. An **instrument setting** says how to
take a reading and stays with the instrument. Sixteen of `branch:`'s thirty-one keys are gates —
`production_min_s`, `voiced_fraction_min`, `f0_spread_max_semitones`, `coverage_min`,
`response_min_s`, `score_min`, `verbatim_overlap_max`, `dominant_segment_min_fraction` among them.
The other fifteen are definitional: `voiced_strength_min` says what *counts* as a voiced frame,
`event_min_s` what the walk will *report*, `pause_min_s` what a gap *is*.

**A gate resolves most-specific-first: family, then group, then default**, a family overriding its
group key by key rather than wholesale. A group naming no value for a gate does not apply it. The
verdict records which layer supplied each gate, so a family-specific bound is legible as one without
opening the config.

The motivating example for that — `GLIDE` being bound by `f0_spread_max_semitones`, a held-vowel
bound on a task whose purpose is that pitch moves — was **true of the config and false of the code**:
`_voice_glide` never called the spread qualifier, so no glide was ever judged by it. The config said
one thing and the code did another, which is its own argument for putting every bound in one
readable table. The layering stands on a case that is real in both: `maximum-phonation-time`
declares `expect_inhale` and its v2 does not, inside one group.

`by_family` and `default` both ship empty, so the packaged config exercises one layer of three.
Every value moved at its current setting: differential replay over the 62,139-recording corpus puts
conformance at `50455/4916/6768` before and after, **zero moved across all 48 families**. Five gate
names are new, each a code literal written down at the value the code already used.

### Two counts, and only one of them may be gated

The counted-breath, cough and loudness families take their number from the instruction: it is in
the task's own name (`fivebreaths`, `threequickbreaths`) or enumerated in `tokens`
(`('hey','hey','hey')`). The diadochokinesis families' 10 and 30 came from nobody — that task asks
for repetition as fast as possible, no number is spoken, and individuals vary.

`expected_event_count` held both. It is replaced by `required_count` and `typical_count`, whose
types say which kind a row carries, what unit it counts in, and whether anything may be judged
against it: a **required** count may be gated, with a tolerance still to be derived; a
**statistically expected** one may never be, and `gates.UNGATEABLE_READINGS` refuses a table that
binds it. No gate reads either today — which costs the DDK families nothing, because a count nobody
asked for should never have decided anything. See
`specs/20260921-required-and-typical-counts/design.md`.

The full design, with the gate/instrument split in full and the measurements each gate reads, is
`specs/20260921-gates-in-verdict/design.md`.

## The gates decide the conformance, keyed by family then group — owner decision, 2026-09-21

> "all gates should be in verdict not in branches and it should be task group related. branches
> should just provide the info necessary for the gates. so instruments sit in preprocess and
> branches."

Until 2026-09-21 the claim above was true of outcomes and false of thresholds: 31 operating points
sat in `branch:` as a flat block applied identically to every task, and the branches applied them.
Sixteen of them are **gates** and are now in `verdict.gates`. Fifteen are **instrument settings** —
what counts as a voiced frame, what the walk will report, what a gap *is* — and stay with the
instrument.

**A bound resolves in three layers, most specific first: `by_family`, then `by_group` keyed by the
`Pattern` each expectation row declares, then `default`.** A task group alone is too coarse:
`SUSTAINED` holds `maximum-phonation-time`, which declares `expect_inhale`, beside its v2, which
does not, and `SYLLABLE_TRAIN` holds two generations of diadochokinesis instruction. A family
overrides its group **key by key** — a family naming one gate inherits every other gate its group
names — and both the family and the default layer ship empty, because every value moved at its
current setting and nothing is universal. **Which layer supplied each bound is recorded**, so a
reader can tell a family-specific bound from an inherited one without opening the config.

**No gate reads either count.** `required_count` may be bound once a tolerance is derived;
`typical_count` may never be, and the gate table refuses it at import. `events_min` and
`repetitions_min` read what the instrument found, at a bound of one.

A branch now writes **no conformance at all**: `Result` carries spans and findings and has no field
for one, and every `branch_report` is written `UNDETERMINED`. This fold reads the gates' inputs off
the branch's own `measure` findings, applies the declared family's group, and substitutes the answer
onto that branch's report. Three rules govern it:

- **A layer that names no value for a gate does not apply it.** `GLIDE` naming no
  `f0_spread_max_semitones` is why a held vowel's pitch bound no longer reaches a sweep.
- **A gate whose reading is absent yields `UNDETERMINED`, never `False`** — and so does a gate whose
  bound nobody has measured, and so does a group that applied no gate at all. An absent reading is
  not a failed one.
- **Only the branch that owns the declared family, and reported `in_family`, is gated.** Every other
  report evaluated no task; QUALITY's conformance is about the store's own assertions and no gate
  reads it.

Each gate applied is recorded on the verdict — the gate, the reading, the value, the bound, the
layer that supplied it and the key it was keyed under, the comparison and the outcome — under
`FileVerdict.gates`, alongside the whole resolved table and the layer of every bound in it. A conformance can be read backwards from the verdict alone. A gate whose
bound is null joins the reporting node's `unmeasured`, so `verdict.unmeasured_points_flag` reaches a
gate the same way it reaches an instrument setting.

Gates whose finding carries an **extent** — a rejected carrier, a located deviation, a per-event
count — keep their bound in the same table and are applied where that extent is known, inside the
reporting node. This fold mints nothing and locates nothing, so it cannot apply them; what it can
do is hold the one number they and it both read. `specs/20260921-gates-in-verdict/` carries the
split, the equivalence argument and the corpus replay behind it.

## The fold is task-aware

> "verdict has to evaluate based on all branches and the task it is assessing."

What a missing conformance *means* is not the same question on a prolonged vowel as on a story
recall. The fold therefore takes the declared task family — resolved by `declared_task` off ADMIT's
recorded path, the same resolution ROUTING wrote onto each decision — and reads every conformance
ground against it. `verdict.conformance_flags_by_family` excepts a family by name; it ships empty,
because excepting one is a claim about that family's expectation row and no such claim is measured.

### DDK is not symmetric with the other three

> "the ddk branch has a specific purpose to detect ddk. the likelihood of a ddk existing by chance is
> close to 0. hence ddk is the one branch where 99.999999% of the time its evaluation is on task."

The other three detect evidence that occurs **incidentally**: breath and cough happen in any
recording, sustained phonation happens in any recording, lexical content happens in any recording. So
for them the in-family and out-of-family modes ask genuinely different questions. A rapid alternating
repetition train does not occur incidentally — for DDK, **finding the subject is evaluating the
task**, and the two-mode symmetry invites reading all four alike.

So for a branch in `verdict.detection_is_evaluation` (`[DDK]`), an **out-of-family** result
contributes no flag ground at all — neither its conformance nor a route mismatch — and is recorded in
`detector_covariates` instead: a fact about the detector's behaviour over the corpus, not a reading of
this recording. In-family it folds exactly like the other three.

The corpus supplied the mechanism, not a number: `ddk.lexical_repetition >= 3` — the gate with no
sweep anywhere — routed DDK on 99% of `rainbow-passage`, 98% of `caterpillar-passage` and 87% of
`free-speech`, all ordinary function-word repetition, while `ddk.ppg_segment_rate_per_s` separated the
two real DDK recordings from every speech recording in the 13-recording sample without overlap. Those
figures are why content must not route DDK, and they are what `routing.declaration_required: [DDK]`
answers. Both gates were removed once it landed, so `branch_gates.DDK` is empty and DDK runs only when
the declaration names it. The out-of-family arm is therefore no longer reachable from the packaged
configuration — see [`dag.md`](dag.md) § *DDK routes on the declaration alone*.
**Nothing here is a weight or a prior.** A per-branch, per-mode prior over the corpus is *owed*; a
number invented to express "much less likely" would be a fit nobody took.

## QUALITY is in the aggregation, and its conformance is about something else

QUALITY is not a routed branch — `run.py` calls it unconditionally on every path PREPROCESS
completed, it has only the out-of-family mode, and it is not in `BRANCHES`. So it has no route, no
`findings` row, no `agreement` row and no `hints` row. What it contributes is a **conformance about
the store's own assertions** (`conformance_of: store_assertions`): `false` when a stored clip
assertion is contradicted by a measurement in the same store, `true` when every checkable clip span
is consistent, and `UNDETERMINED` when nothing was checkable — a recording with no clip span has not
passed an audit, it has had none taken.

That referent is why it always flags on `false`, with no task key and no switch governing it: a
recording whose store contradicts itself is inconsistent whichever task it carries, and no corpus
measurement could make an unclipped sample quieter than a clip ceiling. It is the one conformance in
the graph that owes no ground truth, because both halves of the comparison are the pipeline's own.

**The fold is keyed by branch.** It was keyed by kind until 2026-09-13, when TAXONOMY's kind
classification was deleted; the replacement emits a route state per branch, DDK has no kind at all,
and inventing a second branch-to-kind map to keep the old key would be the parallel vocabulary the
change removed. A branch's own report joins to its decision by node name — `BranchReport.node` **is**
the branch name. The 2026-09-13 rule that a branch verdict with no `kind` was not folded is gone with
the verdict: a report has nothing for a reader to synthesise, because it carries no outcome to be
unreadable.

## Signature

```
verdict(store, hint?) -> file_verdict
```

Writes one element, and it is where every decision about the recording is taken. A branch has
decided nothing for it to defer to: it **combines** what was reported, and where the graph disagrees
with itself it says so rather than resolving the disagreement by precedence.

## Two axes, because they answer different questions

| axis | question | values |
| --- | --- | --- |
| `triage` | what should happen to this recording | `pass` \| `flag` \| `discard` |
| `release` | is a redacted artifact safe to hand on | `releasable` \| `withheld` \| `nothing_to_redact` \| `not_assessed` |

Collapsing them would make a recording with clean measurements and surviving PII look like a
measurement problem, and a recording with an empty transcript and no PII look releasable.

## `discard` is a narrow outcome

**A branch `fail` is not a file `discard`.** A branch `fail` means "this branch has no subject" — a
cough recording has no speech, so SPEECH failing is the expected outcome.

`discard` has exactly two grounds:

| ground | condition |
| --- | --- |
| **unmeasurable** | ADMIT failed. Nothing ran and nothing is claimed about the recording |
| **acoustically empty** | the ruleset's own file-level state is `empty` — the emptiness bypass read every tracked stream peak under `emptiness.peak_floor` — **and no hint claims otherwise** |

The second ground used to be "every kind resolved `absent`", which could never fire while every kind
read `uncertain`; dag.md recorded it as unreachable. It is now a measurement. Its sibling state,
`unexplained` — nothing routed and the recording was *not* empty — adds a ROUTING **flag**, never a
discard: content no gate could account for is a charge against the ruleset, and the two are kept
apart in the product rather than in a reader's head.

**A hint that claims otherwise turns the second ground into a `flag`, never a `discard`.** A file the
graph found nothing in, on which the declaration says there should have been something, is exactly the
case a human must see; discarding it would delete the evidence that the graph was wrong.

The two grounds carry different reasons — "could not measure" and "measured, and there is nothing of
interest in it" — and a consumer that cannot distinguish them treats an empty recording as a broken
one.

## Branch authority is scoped to the branch's own subject

**A branch is the authority on its own subject and on nothing else.** It is the more precise
instrument for what it measures; it is not an instrument for the others.

| the branch's detection | its reach |
| --- | --- |
| SPEECH | lexical speech only. It refutes neither AIRWAY's subject nor VOICE's |
| AIRWAY | airway events only |
| VOICE | phonation only |
| DDK | the repetition train only |

A branch's detection stands in the `findings` map, whatever the ruleset routed. **`findings` is read
off the spans the branch proposed**, not off any conclusion of its own: `present` with one or more,
`absent` where the branch reported and proposed none, `uncertain` where it left no report — not
routed is not a measurement of absence. That is the substantive change of 2026-09-16: `_resolved`
mapped every non-`FAIL` *outcome* to `present`, which let a branch's severity decide its
found/not-found reading; `_found` reads the spans, which are the record. A branch therefore never
raises its own absence to a flag over a declaration — that mismatch is the fold's to name, from the
`hints` table below.

## The route is reported beside the branches, never over them

The route state is written into the product next to the branch conclusions, and the fold records
whether the two **agree** or **mismatch**, per branch. The axis changed with the key: it compared
TAXONOMY's classification against the branch's conclusion, and it now compares the **route** against
it — which is the quantity the ruleset work cares about, countable per branch over a corpus without a
join.

| route | branch found a span | recorded | triage | what it means |
| --- | --- | --- | --- | --- |
| routed | yes | `agree` | — | — |
| routed | no | `mismatch` | — | routing is lenient by design; the branch found none of its kind and was right |
| declined | yes | `mismatch` | `flag` | a miss — the branch ran only because a hint forced it |
| declined | no | `agree` | — | — |
| unavailable | either | `resolved` | — | a branch nothing could judge made no claim to agree with |
| any | branch left no report | `not_run` — see the branch-decision rows below | — | — |

The table is unchanged; only its branch-side input is. It scored the branch's `Outcome`, which was
the branch deciding; it now scores what the branch *found*. Neither side is a judgement, so the
disagreement between them is this fold's to name. The one exception is a branch in
`detection_is_evaluation` run out of family — see the DDK section above.

**Only one direction of a mismatch flags, and it never overrides.** Declined-and-found-it is the
ruleset being wrong about the recording, and flags. Routed-and-found-nothing is not its mirror:
screening is lenient on purpose, so sending VOICE to a read passage is routing working as intended
and VOICE finding no phonation is VOICE being right. Charging the recording for that made it the
largest single flag ground in the corpus — 830 records over a 2,269-recording pilot, 382 recordings
flagged on it and nothing else. Owner-directed 2026-09-20: *it's possible for voice to find nothing.*

Both directions stay in `agreement`, and `findings` and `routes` record the observation on every
recording, so nothing auditable is lost — the fold simply stops deciding on it. The informative
absence is untouched: `hint mismatch` still flags where the recording **declared** a kind the branch
did not find. The routing cannot overturn a branch on its own subject, and the branch does not
rewrite the decision: both stay in the store and both appear in the product, so the disagreement is
visible rather than resolved by precedence.

## A branch that never ran is not a branch that failed

`routing.md`'s `branch_decision` elements are what distinguish the two:

| branch decision | branch report | reading |
| --- | --- | --- |
| `will_run: false`, route `declined`, not forced | none | **expected.** The graph declined to look, and said why |
| `will_run: true` | present | folded as above |
| `will_run: true` | absent | **flag** — the branch was asked and left no answer; the reason names `errored without a verdict`, `completed without a verdict` or `never ran` |
| every branch `will_run: false` | none | the empty execution set — see `discard` above |

The operational states behind that third row — `completed`, `errored`, `skipped` — are the **runner's**
record and are untouched by the report/decide split. `run.py::_attempt` records the first two,
`_drive_branches` the third, and a node that *reported* is `completed` exactly as one that decided is.
They say what the runner did, not what the graph concluded, and nothing in this fold rewrites them.

## Hints are read here, for branch mismatch

The hint's `speech_type` and `may_contain` tags are compared against what the branches concluded.

| case | outcome |
| --- | --- |
| a hinted branch ran and **found nothing** | **`flag`**, naming the mismatch: the branch, and that it was declared and did not find its subject |
| a hinted branch found its subject | recorded as agreement |
| a subject found that no hint claimed | recorded; not a flag on its own |

The claims are read off ROUTING's own record of reading the declaration — the `hint_tags` on each
decision — not re-resolved here against `routing.hint_branch_map`. Two nodes resolving one tag
independently could disagree whenever the config or the hint handed to them differed.

A hint never resolves a subject, never suppresses a branch's conclusion, and never turns a `flag`
into a `pass`. Its one power on this axis is to prevent a `discard` (above) and to name a mismatch.

### `hints` states a falsehood when `hint_branch_map` is null — **owed a code change**

With the packaged `routing.hint_branch_map: null` every declared tag is unmapped, so no decision
carries `hint_tags` and `_hint_claims` returns `{}` rather than `None` (`nodes/verdict.py:182-184`).
The `hints` table is then built (`vocabulary.py:350-357`) through `_hint_reading(claimed=False, …)`
(`vocabulary.py:279-291`), which writes `found_unclaimed` or `no_claim` for **every** branch.

Measured on four runs handed `may_contain: [cough, airway]`: the verdict reads
**`AIRWAY: found_unclaimed`** — an assertion that nobody claimed AIRWAY, on a recording whose
declaration claimed it. It is not confined to the store. It reaches `summary.json`'s
`recording.declared_hints` (`report.py:1097`) and the PDF header (`report.py:1426-1427`, rendered at
`:1467`), verified in a released summary.

`UNREAD_DECLARATION` (`vocabulary.py:385-386`) exists for exactly this case and does not fire:
`_hint_claims` returns `None` only when a hint was supplied and **no decision survived**
(`nodes/verdict.py:182`), not when the decisions survived and could resolve nothing.

**This is a defect, not an owed derivation, and the fix is blocked on a vocabulary decision.** The
product below pins four `hints` tokens; what the field should say when no map is configured — a fifth
token such as `unresolvable`, or `hints: {}` plus the existing flag — is the owner's call and not a
measurement. The upstream causes are registered at
[`../20260913-branch-contract-and-hints/design.md:288-311`](../20260913-branch-contract-and-hints/design.md);
this consequence on the verdict table and the rendered report is registered here and in
[`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § B.

## The triage fold

Evaluated in order; the first that applies wins.

| order | condition | `triage` |
| --- | --- | --- |
| 1 | ADMIT failed | `discard` — unmeasurable |
| 2 | any **deciding** node returned `flag`; any reporting node's conformance was `false` and the policy flags it for this task; any mismatch row above fired; a reporting node named an unmeasured point; the ruleset read `unexplained`; or a branch that was asked to run left no report | `flag` |
| 3 | the ruleset read `empty`, and no hint claims otherwise | `discard` — acoustically empty |
| 4 | otherwise | `pass` |

**With every `branch.*` key shipping a value since 2026-09-16, row 2's conformance ground is
reachable on the packaged config** — it was not while 38 of 39 keys were null and the branches
answered `UNDETERMINED` or raised. An all-`UNDETERMINED` fold is still a sane state and still reads
`pass` on row 4: a question nobody answered is not a finding, and flagging it would flag every
recording no branch was in-family for.

**The graph's stated goal is to be accurate about `pass` and `discard` and to minimise `flag`.** A
fold that flags everything transports no information; a reason that fires on nearly every file is a
reason to re-derive, not a reason to keep flagging.

## The release fold

### REDACT's absence decides nothing — owner decision, 2026-09-22

> "redact is only a function of whether asr exists and contains lexical words, you can't redact
> anything else. so redact not-running is completely fine and should have no implication on
> assessment. that's verdict's job to determine and not dependent on redact."

Until 2026-09-22 `_release_from` read the whole axis off one question — did REDACT leave a verdict?
— and answered `not_assessed` when it had not. But the runner calls REDACT only when
`"SPEECH" in selected and _speech_found_pii(store)`, so its silence is the *ordinary* case, not an
anomaly. That is REDACT deciding by its own absence, which the graph's contract forbids: a branch
reports, VERDICT decides.

**Measured over 62,548 replayed recordings** (`triage_replay_20260922`, probed store by store; see
[the population table](#what-the-old-rule-was-reporting) below), the old rule reported 44,623 —
71.34% of the corpus — as unexamined. Exactly **one** of them was.

### The table

Read in order; the first row that matches wins.

| condition | `release` | `release_ground` |
| --- | --- | --- |
| REDACT left a verdict, `pass` | `releasable`, for **its artifacts only** | — |
| REDACT left a verdict, `flag` or `fail` | `withheld`; unresolved is not cleared | — |
| live `pii` findings and no REDACT verdict | `not_assessed` | `REDACTION_OWED` |
| SPEECH errored | `not_assessed` | `SPEECH_UNREAD` |
| SPEECH left no lexical count and did not complete | `nothing_to_redact` | `NO_TRANSCRIPT` |
| SPEECH completed and left no lexical count | `not_assessed` | `SPEECH_UNREAD` |
| SPEECH read no lexical word | `nothing_to_redact` | `NO_LEXICAL_WORD` |
| the scan was declined — every word is in the stimulus | `nothing_to_redact` | `NOTHING_BEYOND_STIMULUS` |
| the scan ran and found nothing | `nothing_to_redact` | `SCAN_FOUND_NOTHING` |
| SPEECH read lexical words and recorded no scan | `not_assessed` | `SCAN_UNRECORDED` |

The table is total: the last row is the fall-through.

**Why one axis and not two.** "Was there anything to redact" and "did redaction succeed" are
different questions, and collapsing them is what produced the defect — so the split was considered
and rejected. The reason is that the second question does not exist wherever the first answers *no*:
when nothing needed redacting REDACT produces **no artifact at all**, so "did redaction succeed" is
not unanswered, it is not asked. A two-axis product would be a cross-product most of whose cells are
impossible, and the one question a consumer actually has — *may I hand this artifact on?* — would
require joining them. One axis with the states enumerated over the evidence answers it directly;
`release_ground` carries the decomposition without a second axis, exactly as `discard_ground`
already does for triage.

**`nothing_to_redact` is not `releasable`.** There is no artifact, so nothing was cleared. Folding
the two together would assert a clearance for a file that does not exist. This is the old
"`not_assessed` is not `releasable`" warning, kept and narrowed: a recording with no speech, or with
speech and no PII, was never redacted and must not be read as cleared of content a transcript could
not carry.

**`not_assessed` now means only what it says.** Three grounds reach it, and all three are cases
where the graph genuinely could not tell: SPEECH errored, SPEECH reached lexical content and left no
scan record, or a scan found something and no redaction verdict stands over it. The third is a
safety-relevant gap and used to be indistinguishable from the 44,622 recordings that were simply
clean.

**A SPEECH withheld by a critical failure reads `NO_TRANSCRIPT`, not `not_assessed`.** The graph
wanted to run SPEECH and could not, so what the recording carries is genuinely unknown — but there
is no artifact and no transcript either way, so the release axis has nothing to withhold. That
circumstance is already a `CRITICAL_ABSENCE` flag on the **triage** axis, which is the axis that
asks whether a human must look. Keeping it off the release axis is what stops one circumstance from
being counted twice.

**Nothing an annotating detector concluded reaches this axis.** REDACT's optional LLM re-read of the
redacted transcript used to downgrade REDACT's own outcome from `pass` to `flag`, which this table
then read as `withheld` — an unmeasured model gating a release. Corrected 2026-09-17 on the owner's
instruction ("the llm is part of a branch, so it can only annotate (with provenance)"). The re-read's
annotation now reaches the triage axis and only the triage axis; see
[`llm-check.md`](llm-check.md).

**`releasable` never applies to the store.** The store holds the unredacted consensus transcript by
design and is append-only. `release` describes REDACT's artifacts and nothing else.

**The goal on this axis is to minimise `withheld`**: a withhold is a file no consumer can use, and
every withhold that rests on a scan of text nobody uttered is one the graph created.

### The evidence VERDICT reads

`RedactionEvidence`, gathered by `nodes/verdict._redaction_evidence` and nowhere else:

| field | source | meaning |
| --- | --- | --- |
| `lexical_words_n` | SPEECH's `branch_report`, `words_n` | how many lexical words the consensus carried; None where SPEECH left no report |
| `scanned` | every live `pii_scan` measurement | True scanned, False declined, None no record |
| `findings_n` | live `pii` entities | how many findings stand |

This is the owner's rule read literally: *whether asr exists* is `lexical_words_n is not None`
joined to `ran["SPEECH"]`, and *contains lexical words* is `lexical_words_n > 0`. `scanned` only
separates the two ways a transcript with words can carry nothing.

### What the old rule was reporting

Measured 2026-09-22 over every store under `triage_replay_20260922/out/` — 62,548 recordings, read
directly rather than through the differential's aggregate, with the "after" generation recovered
from the invalidation edges the way `replay_diff.split_generations` recovers it. Slurm array
23515058, 64 tasks, `mit_preemptable`; zero read errors.

| old `release` | n | % |
| --- | ---: | ---: |
| `not_assessed` | 44,623 | 71.34 |
| `releasable` | 13,521 | 21.62 |
| `withheld` | 4,404 | 7.04 |

| new `release` | `release_ground` | n | % |
| --- | --- | ---: | ---: |
| `nothing_to_redact` | `NO_TRANSCRIPT` | 19,097 | 30.53 |
| `releasable` | — | 13,521 | 21.62 |
| `nothing_to_redact` | `NOTHING_BEYOND_STIMULUS` | 12,980 | 20.75 |
| `nothing_to_redact` | `SCAN_FOUND_NOTHING` | 11,124 | 17.78 |
| `withheld` | — | 4,404 | 7.04 |
| `nothing_to_redact` | `NO_LEXICAL_WORD` | 1,421 | 2.27 |
| `not_assessed` | `SPEECH_UNREAD` | 1 | 0.00 |

Every movement is `not_assessed -> nothing_to_redact`, 44,622 of them. No recording changes between
`releasable`, `withheld` and anything else, which is the property that makes this safe: the rule
touches only the state REDACT's absence used to produce.

These figures are 31 higher than the differential's on the `NO_TRANSCRIPT` row because the probe
covers all 62,548 stores and the differential compares the 62,516 it could replay; 31 of the 32
not-replayable recordings are ones ADMIT rejected, so SPEECH never ran on them.

### `scanned is None` is three different things

20,519 recordings carry no `pii_scan` measurement, and they are **not one population**. Probed, they
decompose with no overlap and no residue:

| what | n | how it presents |
| --- | ---: | --- |
| SPEECH never ran | 19,097 | no `branch_report`, `ran["SPEECH"] == skipped` |
| SPEECH ran and the consensus carried no lexical word | 1,421 | `words_n == 0`, `diarization == "no_words"`, the note *no consensus word* |
| SPEECH errored | 1 | activity, no report |

All 1,421 of the middle row carry **all three** markers, which identifies them as exactly the
`if not lexical:` early return in `nodes/speech.py` — the one path that reports without reaching
step 7, so it writes no scan record. That is why the fold keys on `lexical_words_n` and not on
`scanned`: the lexical count is written on both paths and distinguishes the three cases directly,
where `scanned is None` conflates them.

Two guards were checked over the same 62,548 and are empty: no recording has lexical words and no
scan record, and no recording completed SPEECH without leaving a report. The `SCAN_UNRECORDED` row
of the table therefore fires on nothing in this corpus and exists so that it cannot fire silently.

## A REDACT `fail` does not flip triage, and no non-pass is invisible

Triage answers whether a human must look at the recording; release answers whether an artifact may be
handed on. A surviving PII finding — REDACT's `fail` — is a release problem, so it does not move the
triage axis. A REDACT `flag` **does** flag the file, by the ladder's row 2 like any other node's flag:
it says verification did not finish, which is a question about the graph rather than about the
artifact.

**It appears in the product regardless.** `reasons` carries REDACT's outcome, its surviving
categories and its `unremediable` set on every non-pass, and the [report](report.md) shows it beside
the branch conclusions. A consumer filtering on `triage == pass` sees the release axis in the same
record and cannot mistake one for the other.

## Product

```
triage:   pass | flag | discard
release:  releasable | withheld | nothing_to_redact | not_assessed
discard_ground: "unmeasurable" | "acoustically_empty" | null
release_ground: one of the seven controlled grounds | null   # null wherever REDACT itself decided
llm_redaction: { status, iterations, flagged, model_id, revision, failure }   # {} when REDACT wrote none
reasons:  [ { node, outcome, kind?, why } ]        # every contributing verdict, in order
ran:      { node: "completed" | "skipped" | "errored" }
branches: { branch: { will_run, forced_by_declaration, route_state, conformance? } }
findings: { branch: "present" | "absent" | "uncertain" }        # read off the spans it proposed
conformance: { node: true | false | "UNDETERMINED" }            # every reporting node, QUALITY included
conformance_of: { node: "task" | "store_assertions" }           # what each conformance is about
deviations: { node: [type, ...] }                               # recorded; folded by nothing
unmeasured: { node: [config path, ...] }                        # what it asked for and nobody measured
detector_covariates: { branch: {conformance, spans_n, route} }  # out-of-family, detection_is_evaluation
declared_family: "prolonged-vowel" | null                       # what every conformance was read against
routes:   { branch: "routed" | "declined" | "unavailable" | "ungated" }   # what the ruleset made of it
route_state: "routed" | "empty" | "unexplained" | null          # what it made of the recording
agreement:{ branch: "agree" | "mismatch" | "resolved" | "not_run" }
hints:    { branch: "claimed_and_found" | "claimed_not_found" | "found_unclaimed" | "no_claim" }
view:     the verdict element id, and the node verdict ids it folded
```

`reasons` carries **every** node's contribution, not only the deciding one.

`discard_ground` names which of the two grounds a `discard` rests on, and is null on every other
outcome. It is in the product because the two grounds are the difference between a broken recording
and an empty one, and a consumer must not have to re-derive that from the reasons.

`findings` and `routes` are both present, always. `findings` is what each branch concluded; `routes`
is what the ruleset made of it. Keeping both is what makes `agreement` checkable by a reader rather
than asserted by this node. `route_state` is null only when ROUTING wrote no evaluation, which is a
reading never made rather than a recording nothing routed.

`ran` comes from two sources and is **merged, the runner's over the store's**: the store derives
`completed` for a node that wrote a verdict, `errored` for one that wrote an activity and no live
verdict, and `skipped` for one that wrote neither; the runner's mapping then overrides per node.
`branches` is the routing decision joined to the branch verdict, so a `skipped` branch carries the
reason it was skipped.

Every read of the store here follows the store's shared rule — an invalidated element is never read,
and of the survivors asserting the same thing the latest write wins, per node for verdicts and per
branch for decisions.

## Out of scope

Ranking recordings, choosing what to do about a flag, overriding a branch on its own subject, and any
threshold that would turn a `flag` into a `pass`.

Derivations live in [`benchmarks/`](benchmarks/).


## Moved from vocabulary.fold_file_verdict (2026-09-20)

Prose cut from the module when rationale was moved out of the code. Recorded here because the
first item is the only written record of a rule the agreement table above does not state, and the
second is the only written record of why a direction is missing from it.

**Only one direction of a route mismatch is a flag ground.** `fold_file_verdict` raises the
mismatch flag where `agreement` is `mismatch` **and** the branch found its kind — the
declined-and-found-anyway row, which is the ruleset being wrong about the recording. A branch
routed to a recording that holds none of its kind found nothing because there was nothing: routing
is lenient by design and the branch is right, so that row is recorded in `agreement` and grounds
no flag. **This contradicts the `routed | no | mismatch | flag | over-routing` row of the agreement
table above, which no code path implements.** One of the two is wrong; nothing here resolves which.

**`UNDETERMINED` contributes no flag ground, and that is load-bearing rather than lenient.** Every
numeric `branch.*` key ships null and `detect_*` evaluates no task, so `UNDETERMINED` is the
packaged answer and flagging it would flag the corpus. Same argument as
`verdict.undetermined_flags` in `config-derivations.md`.

**PREPROCESS and ROUTING are folded from `ran`, not from a verdict entity.** Each is a gate every
later node depends on, and a node that raised wrote no verdict, so a raise there is otherwise
invisible to this fold; a silent, evidence-free `pass` would be a worse outcome than the flag the
fold reports from `ran`.

**The two discard grounds are read off different things.** `unmeasurable` is ADMIT's own fail;
`acoustically_empty` is the ruleset's `empty` state, which is the emptiness bypass having read
every tracked stream peak under its floor. A recording nothing routed that was *not* empty is
`unexplained`, and one whose bypass could not be read at all is `unreadable` — both flag rather
than discard, under their own grounds, because the ruleset failing to account for content and the
run failing to produce the evidence are not the same report.

**Spans are read back from the store, never copied into `BranchReport`.** A count copied into the
report would be a second record able to disagree with the store's.

**`withheld_critical` is the fourth state a branch can be in**, beside routed-and-ran,
ran-and-found-nothing and not-selected; a reader that cannot tell it from not-selected reads a
withheld run as a decision.
