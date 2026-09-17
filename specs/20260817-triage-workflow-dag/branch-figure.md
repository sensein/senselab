# BRANCH FIGURE — the branches' own output, drawn from the store

The sibling of `preprocess_figure`. The code is
`src/senselab/audio/workflows/triage/nodes/figure.py`; nothing here is repeated in it, per
CLAUDE.md.

## What was asked, and what changed in the asking

The first ask was to extend `preprocess_figure` with the branches' outputs. The owner refined it the
same day: *"add another figure similar to the preprocess figure that adds branch outputs to relevant
axes"*. `preprocess_figure` is therefore untouched as a product — it still draws PREPROCESS's and
TAXONOMY's output and nothing else — and `branch_figure` is a second entry point beside it.

## Decision 1 — a sibling in the same module, not a second module

`branch_figure` takes `(store, figure_dir, config, *, run_dir, style, stem)` and returns
`{"figure": …, "branch_summary": …, "page01": …}`, exactly as `preprocess_figure` does, so an
operator who generates one over a completed run directory generates the other the same way.

It lives in `figure.py` rather than a module of its own because everything it shares is
module-private there: `pages`, `FigureStyle`, `_mark_padding`, `_absent_reasons`, `_npz`,
`_continuity`, `_spectrogram_panel`, `_waveform_panel`, `_absent_panel`, `_page_height_ratios`,
`_taxonomy_panel`, `_fit_cell_text`, `_renderer`, `_axis_points_per_second`. A separate module would
have had to either make fourteen privates public or keep a second copy of the page machinery, and a
second copy is how the two products' page widths drift apart.

**What was factored out instead.** Four names in `report.py` were product-neutral store readers
sitting behind a leading underscore, and the branch figure needs all four: `_span_sources` →
`span_sources`, `_initial_span_label` → `initial_span_label`, `_report_entities` →
`report_entities`, `_BRANCH_MEASURES` → `BRANCH_MEASURES`. Renamed outright, no alias, per the
pre-alpha rule. `figure.py` imports `report.py`; the reverse edge does not exist, so there is no
cycle.

**What was deliberately left duplicated.** The page loop itself. `report.py` pages through
`plot_aligned_panels`, whose `time_limits` is validated against the recording's duration
(`plotting.py:549`, `0 <= start < end <= duration`). The figure product's defining convention is
that the final page is **padded** to a uniform width, which puts the page's right edge past the
duration by construction. Reusing `plot_aligned_panels` would therefore have meant either
abandoning the padded page or relaxing a validation that `report.py` relies on. Neither was worth
it, so `branch_figure` draws with matplotlib directly in `figure.py`'s own idiom and the page loop
is the one duplicated thing.

## Decision 2 — the relevant axes

Five panels, top to bottom:

1. **wideband spectrogram** — a proposal is a claim about the signal; a lane of bars over nothing is
   unreadable.
2. **waveform, envelope, floor and continuity** — the *initial* spans in every lane below are
   PREPROCESS's amplitude spans, so the reader needs the amplitude axis those were cut from in
   order to judge the pairing at all.
3. **one lane per branch**, in `BRANCHES` order, each drawing its own family as `BRANCH_FAMILY`
   binds it: AIRWAY → `airway`, SPEECH → `speech`, VOICE → `voice`. DDK was dissolved into SPEECH,
   and its syllable spans mint `family="speech"` with `role` and `production` distinguishing them,
   so they land in the SPEECH lane with no special case.

## Decision 3 — the pairing is the edge, never the overlap

Each lane is two rows: **proposed** below, **initial** above, joined by a connector.

`propose_span` (`branches.py:348`) is the only writer of a branch span. It stamps `family` and
`role`, and it refuses a proposal that names no evidence — twice, once in `proposer`
(`branches.py:218`) and again at the write (`branches.py:364`). So *every* branch span carries a
family, a role and at least one `wasDerivedFrom`. That is what makes a single uniform lane correct
for all three branches.

The index is `report.span_sources`, the same one `report.py`'s `_derived_lane` uses, built once per
figure. It keeps only parents that are **live spans with an extent**, so a derivation naming a
measurement, a word or a speaker contributes no initial row rather than a fabricated one.

A connector is drawn only where both ends are on the page — the rule the shared token renderer
already applies (`plotting.py:1097`: "a `derived_from` entry naming a key no token on this page
carries draws nothing").

This is the defect the owner named: `report.py` had looked up `_spans_of_family(store, "phonation")`
— a role in the family field — and had walked AIRWAY assertions for a verb AIRWAY never writes. An
overlap-based pairing is the same class of error one level up: it produces a plausible picture that
is not the graph. `TestThePairingFollowsTheDerivationEdge::test_the_parent_is_the_one_named_not_the_one_overlapped`
puts a proposal wholly inside one envelope span while naming another, so the two rules disagree by
construction; an overlap implementation fails it.

## Decision 4 — four run states, not two

The brief asked that a branch which did not run render distinguishably from one that ran and found
nothing. The store actually supports four, and collapsing any pair would lose a fact:

| state | what the store holds | what the lane says |
|---|---|---|
| `ran` | a `branch_report` exists | the bars, or *ran and proposed no `<family>` span* |
| `withheld` | a `branch_decision` with `will_run` false | *did not run — route `<state>`: `<why>`* |
| `asked, no report` | `will_run` true, no `branch_report` | *was selected to run and wrote no report* |
| `undecided` | no `branch_decision` at all | *ROUTING wrote no decision for `<branch>`* |

`ran` is keyed to the **report**, not to the route: on the fixture recording ROUTING marks AIRWAY
`unavailable` and AIRWAY reports anyway, and reading the route instead would have called that not
run. The third state is the one that would otherwise be silently folded into "withheld"; a branch
that was asked and raised is neither withheld nor found-nothing.

## Decision 5 — the branch reports go on the cover

Conformance, `conformance_of`, deviations, unmeasured points and every `BRANCH_MEASURES` value are
**file-scoped**: they are the branch's reading of the whole recording. Printed inside a page they
would sit in a twenty-second frame and be read as a measurement of that window — the same category
error the config derivations already rule on for whole-file scores used as per-window thresholds.
So the full block is on the cover, and `branch_summary.json` carries it verbatim beside the PDF, as
`taxonomy_summary.json` does for the other product.

What each page needs in order to read its own lanes is smaller: whether the branch ran, and its
conformance. That goes in the lane's own title, which is page-local by construction and is one line.

`BRANCH_MEASURES` is read with `key in attributes`, not `attributes.get(key) is not None`. SPEECH
writes `nontarget_speech_s=None` on a run with no lexical word (`speech.py:1427`) and omits the
seventeen syllable keys entirely unless the declared family is a syllable repetition and the mode is
align (`speech.py:1394`). "Written as None" and "never written" are different facts and the block
keeps them apart.

## What the drawing values are

Six new `FigureStyle` fields, and nothing anywhere else: `branch_height_ratios`,
`colour_branch_initial`, `colour_branch_proposed`, `colour_branch_link`, `branch_row_height`,
`branch_link_linewidth`. The two fills are `report.py`'s own `_INITIAL_FILL` and `_PROPOSED_FILL`,
and the row names are its `_INITIAL_ROW` and `_PROPOSED_ROW`, so one pairing reads the same in both
products; a test pins that equality rather than leaving it to drift.

## Names read that nobody writes

Every attribute, entity kind, measurement name and family value the new code reads was checked
against its writer before it was read. Four findings came out of doing the same sweep over the
existing `figure.py`, and one over `ddk.py`.

### 1. `figure.py` read a `detail` attribute no writer produces — fixed

`_absent_reasons` did `entity.attributes.get("detail").get("absent")`. `write_verdict`
(`common.py:194`) splats its `detail` into the attributes: `{"node": …, "outcome": …, "kind": …,
"why": …, **detail}`. PREPROCESS passes `detail={"absent": …, "derivatives": …}`
(`preprocess.py:3197`), so the stored attribute is `absent`, at the top level. Nothing in
`src/senselab/` writes an attribute literally named `detail`. `report.py:1141` reads it correctly.

`_absent_reasons` therefore returned `{}` on **every real run**, and every name downstream of it was
dead: `energy_envelope`, `normalized_envelope`, `continuity_trace`, `consensus_transcript`,
`spectrogram_wideband`, `span_yamnet`, `span_hear`, `residual`, `{yamnet,ast,hear}_scores`, and the
twelve `{enhanced,residual}_{classifier}` keys. Every absent panel and every absent classifier block
printed a hardcoded note of the figure's own instead of the exception PREPROCESS recorded — which
is precisely what the module docstring says it does not do.

It also took the **first** PREPROCESS verdict rather than the latest, against the store's shared
latest-wins read rule, so after an extend pass it would have reported the superseded run's absences.

Both fixed. The fixture that kept it green (`span_raster_test.py:71`) hand-built
`{"node": "PREPROCESS", "detail": {"absent": …}}` instead of going through `write_verdict`; it now
goes through the writer, as every real producer does.

One consequence is now visible that was not before: `_columns` pads each classifier block to
`_SUMMARY_COLUMN_WIDTH = 52` characters, so a recorded reason longer than that is truncated on the
cover. The key name survives in the cases measured, and the untruncated text is in
`taxonomy_summary.json`, so this is recorded rather than changed.

### 2. `figure.py` duplicated `SUMMARISED_CLASSIFIERS` by value — fixed

`_SUMMARISED_CLASSIFIERS` restated `taxonomy.SUMMARISED_CLASSIFIERS`. `taxonomy` iterates its own
when writing `{classifier}_label_summary`, so the two agreeing is what makes the figure's lookups
resolve, and nothing enforced it. Now imported.

### 3. `_stream_path` was dead — removed

No caller in `src/senselab/` or `src/tests/`. `preprocess_figure` resolves the stream itself.

### 4. `ddk.lexical_repetitions_n` is always zero — reported, not fixed

`ddk.py:1105` counts `component.role == "lexical_repetition"`. No writer mints a span with that
role: the only producer of a lexical repetition is `ddk.py:1010`, which mints `role="task_extent"`
and puts `lexical_repetition` in the **`production`** attribute. `"lexical_repetition"` appears in
`nodes/` only at `ddk.py:1014` (the attribute) and `ddk.py:1105` (the counter).

Two consequences, in opposite directions. The count is always 0. And because the span's role *is*
`task_extent`, and `TRAIN_ROLES = ("task_extent", "repetition")` (`ddk.py:90`), a lexical repetition
is counted as a syllable train in `trains_n` and its duration lands in `train_s` — against that
constant's own docstring, which says a lexical repetition is "a repeated word, not a syllable
train". Separately, no writer mints `role="repetition"` either, so that half of `TRAIN_ROLES` is
also unreachable.

This is a defect in **what the pipeline computes**, not in what a figure draws, and the brief for
this change forbids a pipeline write to make a figure read better. Left alone and recorded here. The
branch figure renders the field faithfully: it prints the 0 the branch reported, because printing
anything else would be the figure deciding.

## What drives this product

Nothing, and that is unchanged from `preprocess_figure`, which `dag.md` already records as being
called by no caller outside its tests. Both are re-invocable over a completed run directory by hand,
which is how the owner generates them on the cluster. Wiring either into `GRAPH_ORDER` or
`scripts/triage_audio.py` is the owner's call, not this change's: `triage_audio.py` is a
two-argument CLI by deliberate design, and adding a figure to every production run is a per-run cost
nobody has asked for.
