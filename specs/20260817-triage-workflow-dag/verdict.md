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
| conformance `false` | the one claim a branch makes about the recording against what the instruction asked for | yes, gated on the referent and the declared family |
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
2026-09-16, which of the eleven declared types that argument actually fits and which are genuine
departures awaiting a measured rate. The constraint survived
this change and is now a declared switch — `verdict.deviation_flags`, shipping `false` — rather than
an implicit rule in the code, so that flipping it is a visible decision with a derivation.

**Every threshold that turns a reading into a judgement is in the `verdict:` config section**, not in
`branch:`. Three keys moved there on the same day for that reason: `min_contrast_db`,
`tilt_max_db_per_octave` and `level_min_dbfs`, all three whole-recording judgements and all three
still unset. `config-derivations.md` § verdict carries every value's derivation.

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
| `release` | is a redacted artifact safe to hand on | `releasable` \| `withheld` \| `not_assessed` |

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
| routed | no | `mismatch` | `flag` | over-routing |
| declined | yes | `mismatch` | `flag` | a miss — the branch ran only because a hint forced it |
| declined | no | `agree` | — | — |
| unavailable | either | `resolved` | — | a branch nothing could judge made no claim to agree with |
| any | branch left no report | `not_run` — see the branch-decision rows below | — | — |

The table is unchanged; only its branch-side input is. It scored the branch's `Outcome`, which was
the branch deciding; it now scores what the branch *found*. Neither side is a judgement, so the
disagreement between them is this fold's to name. The one exception is a branch in
`detection_is_evaluation` run out of family — see the DDK section above.

**A mismatch flags; it never overrides.** The routing cannot overturn a branch on its own subject,
and the branch does not rewrite the decision: both stay in the store and both appear in the product,
so the disagreement is visible rather than resolved by precedence.

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

| condition | `release` |
| --- | --- |
| REDACT did not run — no speech branch, no words, or no PII found | `not_assessed` |
| REDACT returned `fail` — a finding survived verification | `withheld` |
| REDACT returned `flag` — verification was incomplete | `withheld`; unresolved is not cleared |
| REDACT returned `pass` | `releasable`, for **its artifacts only** |

Only `pass` clears an artifact, which makes the mapping total.

**Nothing an annotating detector concluded reaches this axis.** REDACT's optional LLM re-read of the
redacted transcript used to downgrade REDACT's own outcome from `pass` to `flag`, which this table
then read as `withheld` — an unmeasured model gating a release. Corrected 2026-09-17 on the owner's
instruction ("the llm is part of a branch, so it can only annotate (with provenance)"). The re-read's
annotation now reaches the triage axis and only the triage axis; see
[`llm-check.md`](llm-check.md).

**`releasable` never applies to the store.** The store holds the unredacted consensus transcript by
design and is append-only. `release` describes REDACT's artifacts and nothing else.

**`not_assessed` is not `releasable`.** A recording with no speech, or with speech and no PII, was
never redacted, and must not be read as cleared of content a transcript could not carry.

**The goal on this axis is to minimise `withheld`**: a withhold is a file no consumer can use, and
every withhold that rests on a scan of text nobody uttered is one the graph created.

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
release:  releasable | withheld | not_assessed
discard_ground: "unmeasurable" | "acoustically_empty" | null
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
