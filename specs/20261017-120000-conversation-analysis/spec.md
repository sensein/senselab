# Conversation analysis over word-timed, diarized transcripts

2026-10-17. Module: `src/senselab/text/tasks/conversation_analysis/`. Entry point
`analyze_conversation(words, diarization=None, config=None) -> ConversationAnalysis`.

This file is where the definitions, the reasoning behind each rule and the references live; the code
carries only what each function is and how to call it. Thresholds are in `data/defaults.yaml`, each
with a short derivation; lexicons are in `data/lexicons.yaml`. **No threshold has been fitted to
labelled senselab data.** Every value is a literature convention or the task brief's value; a later
measurement should regenerate the profile, not hand-edit it.

The YAML files are read with YAML 1.2 booleans (only `true`/`false`). The repository's
`pretty-format-yaml` hook writes YAML 1.2 and strips the quotes from `yes`, `no`, `on`, `off`; PyYAML's
default YAML 1.1 resolver would then read those lexicon words as booleans and silently drop "yes" from
the backchannel lexicon. `test_yaml_boolean_words_load_as_strings` pins this, and the loader refuses any
non-string lexicon entry.

## Scope and placement

The consumer is a pipeline analysing two-person Zoom interviews, but every computation is n-party and
treats speakers as opaque labels. The input is what ASR + diarization already produce: words with
`speaker`, `start`, `end`, `text`, optional `confidence`. No audio, model or network is touched, so the
module sits under `text/tasks/` (beside `pii_detection`, which also works over `ScriptLine` words)
rather than `audio/tasks/`. Acoustic features enter only through the generic entrainment helper, as
per-turn numbers computed elsewhere.

Prior art: commit `56af342a` (branch `codex/conversation-analysis-features`) added an
`audio/workflows/conversation_analysis.py` that ran diarization, ASR, features, emotion and embeddings
and attached lexical cues per diarized segment. Reused here: the cue lexicons (backchannel cues,
discourse markers, politeness cues, question/request/command prefixes) as seeds, and the idea of a
per-speaker summary plus per-turn table. Not ported: the model orchestration, recording QC and the
segment-as-turn assumption — a diarization segment is not a turn (it splits at pauses and does not
know about backchannels), which is why turns here are built from IPUs.

## Input normalisation

* Words are flattened (a `ScriptLine` with `chunks` contributes its chunks, which inherit its speaker),
  sorted by `(start, end, input position)` and numbered. Order of input does not matter.
* A word without `start`/`end`, or with `end < start`, is an error, not a silent drop. Zero-duration
  words are allowed (aligners emit them).
* A word without a speaker is attributed from `diarization`: the segment overlapping it most, ties by
  distance from the word midpoint. Without diarization it is an error.
* Whitespace-only words are skipped. **Nothing else is dropped or rewritten.** In particular fillers
  (`uh`, `um`, CrisperWhisper's `[UH]`, `[UM]`) are kept verbatim in `Word.text`, IPU/turn text and
  every table; `Word.token` is a normalised copy (lower case, curly apostrophes folded, edge punctuation
  stripped, brackets and a trailing cut-off hyphen kept). Fillers are individual markers (Clark & Fox
  Tree 2002), and CrisperWhisper (Wagner et al. 2024) exists precisely to transcribe them; a
  normalisation step that stripped them would silently zero every filler metric. The test
  `test_fillers_survive_end_to_end` pins this.

## Turn-taking

### IPU (inter-pausal unit)
A maximal run of one speaker's words in which no silence between consecutive words reaches
`ipu_gap_s` (0.2 s). Silence is measured from the running maximum `end` of the unit, so overlapping
words of one speaker never split it. 200 ms is the conventional talkspurt/pause boundary (Brady 1968;
Heldner & Edlund 2010) and sits above stop-closure silences. Gravano & Hirschberg (2011) use 50 ms on
hand-aligned words; with ASR word times 50 ms would split at alignment jitter.

### Backchannel
An IPU is a backchannel when all hold:
1. **short**: duration ≤ `backchannel_max_duration_s` (1.0 s) and ≤ `backchannel_max_words` (3);
2. **lexically backchannel-like** (its tokens segment entirely into entries of the backchannel lexicon,
   so "oh okay", "yeah yeah" and "i see" qualify) **or fully overlapped** (its span lies inside the union
   of other speakers' IPUs);
3. **does not take the floor**: another speaker's IPU is still running at its end or starts within
   `backchannel_floor_window_s` (1.0 s) after it;
4. **is not an answer**: the latest other-speaker IPU ending before it does not end in `?`.

Rule 4 was added for interviews: a bare "Yes." to "Do you smoke?" is short, lexical and the
interviewer resumes quickly, so rules 1–3 alone would call every one-word answer a backchannel. It
depends on ASR punctuation; without punctuation answers to yes/no questions can still be
misread as backchannels (open question below).

`backchannel_type` is `overlapping` if the IPU intersects any other speaker's IPU, else `in_pause`.
Backchannel definitions follow Yngve's "back channel" and the operationalisation of Ward & Tsukahara
(2000) and Heldner, Hjalmarsson & Edlund (2013): short, non-floor-taking, mostly lexicalised responses.

Backchannel IPUs are not part of any turn, and their words are not scanned for disfluencies: a
listener's "hm" continuer is not that listener's filled pause.

### Turn
Non-backchannel IPUs are visited in start order with one open floor-holding turn:
* an IPU by the floor holder extends the turn (so a backchannel by the other speaker does not split it);
* an IPU by another speaker that lies wholly inside the current turn's span becomes an **embedded**
  turn — it is talk, but the floor never changed hands (an unsuccessful interruption, a collaborative
  completion, a side remark);
* any other IPU opens a new floor-holding turn.

This is the brief's "consecutive IPUs of the same speaker merged unless the other speaker produced a
non-backchannel IPU in between", with "in between" read as "took the floor". Read literally on start
order, an objection inside a long answer would split the answer into two turns and create a spurious
transfer whose FTO is the length of the objection.

### Floor transfer offset (FTO)
For consecutive floor-holding turns with different speakers, `fto = next.start − previous.end`
(negative = overlap). `|fto| < no_gap_no_overlap_s` (50 ms) is `no_gap_no_overlap`; otherwise
`gap` (positive) or `overlap` (negative). Definitions after Heldner & Edlund (2010) and Levinson &
Torreira (2015); the modal FTO in conversation is ~200 ms (Stivers et al. 2009). The 50 ms band is about
two frames of the 20–40 ms resolution of word aligners.

`after_question` marks transfers whose yielding turn's **last** dialogue-act segment is a question
(wh, yes-no, declarative/tag); the taker's FTO on those transfers is the response latency after
questions.

### Overlap episodes
Every pair of IPUs of different speakers with positive-length intersection is one episode
`[later.start, min(ends)]`. The holder is the speaker whose IPU started first. Classes:
* `backchannel` — either IPU is a backchannel;
* `non_competitive` — onset within `terminal_overlap_window_s` (0.5 s) of the end of the holder's IPU:
  a terminal overlap at a transition-relevance place (Jefferson 1984; Schegloff 2000);
* `competitive` — otherwise (an interruption). `successful` when the holder's IPU ends within
  `interruption_yield_window_s` (2 s) of the onset and the interrupter's IPU outlasts it; otherwise
  `unsuccessful`.

Episodes are IPU-level, so a sub-200 ms within-IPU pause counts as speech. With three or more speakers
pairs are scored independently.

### Per-speaker turn-taking summary
* `speaking_time_s` — sum of the speaker's IPU durations (backchannels included); `speaking_share` —
  over the sum for all speakers (simultaneous speech counted for both).
* `n_turns` (all), `n_floor_turns` (non-embedded), `turns_per_min` over the conversation span (first
  word start to last word end).
* turn duration and words: mean, median, sample SD (ddof = 1; `None` below two turns);
  `longest_monologue_s`/`_words` = the longest turn.
* `n_backchannels` (+ overlapping / in-pause), `backchannels_per_min`.
* interruptions made / received, each split successful / unsuccessful (competitive overlaps only).
* `floor_holding_index` — of the competitive overlaps the speaker took part in, the fraction they won:
  `(made_successful + received_unsuccessful) / (made + received)`. Other uses of the name exist
  (e.g. share of floor time); this one measures overlap resolution, which is what distinguishes a
  dominant speaker from a talkative one.
* `n_floor_takings`, `fto_mean_s`, `fto_median_s`, `prop_gap`, `prop_overlap`,
  `prop_no_gap_no_overlap` — over transfers *to* the speaker.
* `response_latency_after_question_mean_s` / `_median_s`, `n_responses_after_question`.
* `diarized_speaking_time_s` when diarization was given.

## Disfluency

Scanned over turn words only (not backchannel IPUs). Taxonomy after Shriberg (1994): filled pauses,
repetitions, cut-offs (false starts), prolongations. Rates are per 100 of the speaker's turn words and
per minute of the speaker's turn speaking time (sum of the durations of IPUs in their turns).

* **Filled pause** — token in the filler lexicon after dropping brackets and a trailing hyphen,
  case-insensitively (`[UH]`, `Um,`, `uhm`). Families for the uh:um ratio (Clark & Fox Tree 2002):
  `uh` {uh, uhh, er, err}, `um` {um, umm, uhm, erm}; `other` {ah, eh, hm, hmm, mm} counts as a
  filled pause but enters neither side of the ratio. `uh_um_ratio = n_uh / n_um` (`None` without um).
  Clark & Fox Tree argue *uh* signals a minor and *um* a major upcoming delay, so the ratio and the
  pause after each are reported separately rather than merged.
  Per filler: `position` turn_initial / turn_medial / turn_final / turn_isolated (sole word);
  `pause_before`/`pause_after` = silence to the speaker's previous/next word within the turn;
  `pre_pause` = followed by a silent pause ≥ `silent_pause_min_s` (0.2 s, the IPU boundary),
  `post_pause` = preceded by one; `next_event` = `pause`, `word`, `speaker_change` (turn-final and
  another speaker talks after it) or `end`. Summary: counts per position, `fillers_pre_pause`,
  `fillers_post_pause`, mean/median duration, mean pause before/after, `p_filler_to_pause`,
  `p_filler_to_speaker_change` (fractions of all the speaker's fillers).
* **Repetition** — over the turn's non-filler tokens (so "I uh I" is a repetition), an n-gram
  (n = 3, 2, 1, longest first) immediately repeated. One event per repeat, covering both copies; scanning
  resumes after the first copy, so "the the the" is two events. Repeats made only of tokens in
  `repetition_exempt` ("yeah yeah", "very very", "no no") are emphatic or backchannel reduplication, not
  disfluency.
* **Cut-off** — a token ending in `-`, `—` or `–` after at least one letter ("wh-", "basi—"); fillers
  excluded. Partial words without a hyphen cannot be detected without a dictionary and are not.
* **Prolongation** — per speaker, `duration / letters` for timed non-filler turn words; z-score with
  the population SD; `z > prolongation_z` (2) is a prolongation. Needs ≥ `prolongation_min_words` (10)
  words (with n words the largest attainable z is √(n−1)). Normalising by letters is a crude stand-in
  for phone count; word-final lengthening at phrase ends is not separated from disfluent prolongation.

## Dialogue acts (rule layer)

Segments: each turn split after words ending in `.`, `?`, `!`; a turn with none is one segment. Each
backchannel IPU is its own segment (`turn_id=None`), labelled `backchannel` (0.9). Classes are a coarse
collapse of SWBD-DAMSL (Jurafsky et al. 1997; Stolcke et al. 2000): wh_question, yes_no_question,
declarative_question (incl. tag questions), directive, backchannel, agreement, disagreement, thanking,
apology, greeting_closing, statement.

Rules, first match wins, on the segment's tokens with fillers removed and leading
discourse tokens (`lead_skip`: so, well, okay, oh, um ...) skipped for head-word tests:
1. head is a wh-word and the segment ends in `?` (0.9) or is unpunctuated (0.6); or ends in `?` with a
   wh-word later and no auxiliary head (0.7) → wh_question. A wh-head ending in `.` ("What I mean is…")
   is a statement.
2. head phrase in `directive_phrases` (could you, can you, would you, I'd like you to, let's …) or
   head "please" → directive (0.8). The brief puts "could/can you" with directives; "Can you hear me?"
   is therefore a directive, not a yes-no question.
3. auxiliary head + `?` (0.9) or unpunctuated (0.6) → yes_no_question.
4. imperative base verb head, not a question → directive (0.7); "please" anywhere with > 2 tokens →
   directive (0.7) ("yes please" stays agreement).
5. ends in `?` → declarative_question (0.8).
6. thanking (0.9), apology (0.9), greeting/closing (0.85), disagreement (head "no", "I disagree" …;
   0.7 short / 0.6 long).
7. ≤ 3 tokens composed of backchannel entries → agreement (0.7) if the latest other-speaker act was a
   question and the tokens are agreement tokens, agreement (0.6) if they are agreement phrases
   ("exactly"), else backchannel (0.7). A filler-only segment composed of backchannel entries ("Hmm.") is
   a backchannel (0.5).
8. agreement phrase ("that's right", "I agree") → agreement (0.6).
9. otherwise statement (0.3).

`confidence` is the rule's strength, ordered by how often such a cue misfires in our judgement; it is
not a calibrated probability.

### A turn's primary act

`primary_act(segments, config)` fixes a turn's `dialogue_act` in three steps:

1. The turn is **substantive** when any segment labelled a question (wh / yes-no / declarative),
   directive or statement holds at least `substantive_min_content_words` (2) content words — tokens in
   neither the `stopwords` nor the filler lexicons.
2. In a substantive turn, `backchannel` and `agreement` are not candidates.
3. The remaining candidates rank questions > directive > disagreement > agreement > apology >
   thanking > greeting/closing > backchannel > statement; acts outside the rule taxonomy (custom
   labellers) rank after, first seen wins.

So "Thanks for coming. What brings you here?" is a wh_question, and "Sure. I work as a nurse." is a
statement. Before step 2 existed, backchannel and agreement outranked statement unconditionally and a
downstream pipeline on two-person interviews saw answers opening with "Sure." / "Yeah." counted as
backchannel or agreement turns.
A bare "Sure. Yes I do." (0 content words) still reads as a minimal
response. Thanking, apology and greeting/closing still outrank a statement: they are not minimal
responses, and "Thanks. I'm fine." is plausibly a thanking turn; that is a choice, not a measurement,
and is the next thing to revisit if a labelled set shows otherwise. The threshold 2 is a convention:
"yes I do" and "that's it" have 0 content words, "oh I see" 1, and an echo of the partner
("Sure. Nursing.") 1; the smallest predicate+argument clause ("I work as a nurse" → work, nurse) has 2.

**Pluggable**: `label_dialogue_acts(segments, labeller=None)` accepts any callable mapping the segment
sequence (in time order, so context is available) to one `DialogueActLabel(act, confidence, source)`
per segment. A local-LLM labeller can therefore be added downstream without touching this module;
acts outside the rule taxonomy flow through to the tables and per-speaker `act_<name>` counts.

## Engagement and politeness markers

Per turn (`TurnMarkers`), summarised per speaker per 100 turn words:
* **discourse markers** (Schiffrin 1987): so, well, you know, I mean, like, actually, anyway, basically,
  right, okay, now, then, because, but, and, also. Multi-word entries matched longest-first. Lexical
  only: "like" as a verb counts.
* **hedges** (Lakoff 1973; Prince et al. 1982): maybe, perhaps, I think, sort of, kind of, I guess,
  probably, …
* **politeness strategies**, the lexical subset of Danescu-Niculescu-Mizil et al. (2013): gratitude,
  apologizing, please (non-initial), please_start, indirect_btw ("by the way"), direct_question (segment
  starts with what/why/who/how), direct_start (so/then/and/but/or), counterfactual_modal (could/would
  you), indicative_modal (can/will you), first_person_start, first_person (non-initial),
  first_person_plural, second_person (non-initial), second_person_start, hedges, factuality ("in fact",
  "actually", …), deference (segment starts with great/good/nice/…), greeting, positive_lexicon,
  negative_lexicon (small built-in lists; the original used Liu et al.'s opinion lexicon). Start-anchored
  strategies count segments; the rest count occurrences. The original features are per request; here
  they are counted over every segment of a turn, so they describe a speaker's style rather than
  request politeness.
* **questions asked** and `questions_per_min` (segments with a question act).
* **pronouns** I / we / you (with contractions), per 100 words (Pennebaker-style function-word rates).
* **lexical alignment** (Pickering & Garrod 2004): distinct content tokens of a turn (not stopwords or
  fillers) that also occur in the latest earlier turn by another speaker, divided by the turn's distinct
  content tokens. Lowercased tokens stand in for lemmas: no lemmatiser is a dependency, so "hike" and
  "hiking" do not align.

## Entrainment

`compute_entrainment(turn_features, features=None, speaker_col, time_col, n_surrogates, seed)` on any
per-turn numeric series (mean f0, intensity, speech rate …), following Levitan & Hirschberg (2011).
Adjacent pairs are consecutive rows (by time) with different speakers; `y_{t-1}` is the partner's turn,
`x_t` the responding turn. Per feature, pooled (`all`) and per responding speaker:
* **proximity** — mean over pairs of `mean_j |x_t − y_j|` over the partner's non-adjacent turns minus
  `|x_t − y_{t-1}|`. L&H sample ten non-adjacent turns; using all of them makes the value
  deterministic. Positive = local similarity beyond the speakers' global similarity.
* **convergence** — Pearson r of `|x_t − y_{t-1}|` with turn time; negative = becoming more similar.
* **synchrony** — Pearson r of `x_t` with `y_{t-1}`.
* **surrogate baseline** — `n_surrogates` random re-pairings of partner values (proximity, synchrony)
  or permutations of differences against time (convergence); reported as the surrogate mean and a
  permutation p-value `(1 + #{surrogate ≥ observed}) / (n + 1)` (one-sided for proximity, on |r| for
  the correlations). Correlations need ≥ 3 pairs and non-constant series, else `None`.

`analyze_conversation(..., turn_features=df)` merges a frame keyed by `turn_id` onto the floor-holding
turns (embedded turns break adjacency and are excluded) and stores the results on
`ConversationAnalysis.entrainment`.

## Outputs

`ConversationAnalysis` holds typed lists (`Word`, `IPU`, `Turn`, `FloorTransfer`, `Overlap`,
`Disfluency`, `DialogueActSegment`, `TurnMarkers`, `EntrainmentResult`), `.to_dataframes()` (words,
ipus, turns, overlaps, floor_transfers, disfluencies, dialogue_acts, markers with politeness flattened
to `politeness_<strategy>`, summary, entrainment) and `.summary()` (`{speaker: {metric: value}}`,
`None` where undefined). Empty input yields empty frames with their columns.

## Open questions

1. Rule 4 of the backchannel definition depends on ASR punctuation. With unpunctuated ASR a one-word
   answer to a yes/no question will be a backchannel. A prosodic or DA-model check is the better fix.
2. None of the thresholds is measured on Zoom interviews. Zoom adds 100–300 ms of transport latency
   per direction that is *not* in a single mixed recording but is in two separately recorded local
   channels; FTO distributions from the two set-ups are not comparable without a latency correction.
3. Lexicons are English only.
4. Embedded-turn detection uses the floor turn's span at the time the IPU arrives; a long overlap that
   ends exactly with the holder's IPU is embedded, one that ends a frame later takes the floor.

## References

* Brady, P. T. (1968). A statistical analysis of on-off patterns in 16 conversations. *Bell System
  Technical Journal*, 47(1), 73–91.
* Clark, H. H., & Fox Tree, J. E. (2002). Using *uh* and *um* in spontaneous speaking. *Cognition*,
  84(1), 73–111.
* Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A
  computational approach to politeness with application to social factors. *ACL 2013*.
* Gravano, A., & Hirschberg, J. (2011). Turn-taking cues in task-oriented dialogue. *Computer Speech &
  Language*, 25(3), 601–634.
* Heldner, M., & Edlund, J. (2010). Pauses, gaps and overlaps in conversations. *Journal of Phonetics*,
  38(4), 555–568.
* Heldner, M., Hjalmarsson, A., & Edlund, J. (2013). Backchannel relevance spaces. *Nordic Prosody XI*.
* Jefferson, G. (1984). Notes on some orderlinesses of overlap onset. In *Discourse Analysis and
  Natural Rhetoric*, 11–38.
* Jurafsky, D., Shriberg, E., & Biasca, D. (1997). *Switchboard SWBD-DAMSL shallow-discourse-function
  annotation coders manual*. University of Colorado.
* Lakoff, G. (1973). Hedges: A study in meaning criteria and the logic of fuzzy concepts. *Journal of
  Philosophical Logic*, 2, 458–508.
* Levinson, S. C., & Torreira, F. (2015). Timing in turn-taking and its implications for processing
  models of language. *Frontiers in Psychology*, 6, 731.
* Levitan, R., & Hirschberg, J. (2011). Measuring acoustic-prosodic entrainment with respect to
  multiple levels and dimensions. *Interspeech 2011*.
* Pickering, M. J., & Garrod, S. (2004). Toward a mechanistic psychology of dialogue. *Behavioral and
  Brain Sciences*, 27(2), 169–190.
* Schegloff, E. A. (2000). Overlapping talk and the organization of turn-taking for conversation.
  *Language in Society*, 29(1), 1–63.
* Schiffrin, D. (1987). *Discourse Markers*. Cambridge University Press.
* Shriberg, E. E. (1994). *Preliminaries to a theory of speech disfluencies*. PhD thesis, UC Berkeley.
* Stivers, T., et al. (2009). Universals and cultural variation in turn-taking in conversation. *PNAS*,
  106(26), 10587–10592.
* Stolcke, A., et al. (2000). Dialogue act modeling for automatic tagging and recognition of
  conversational speech. *Computational Linguistics*, 26(3), 339–373.
* Wagner, L., Thallinger, B., & Zusag, M. (2024). CrisperWhisper: Accurate timestamps on verbatim
  speech transcriptions. *Interspeech 2024*.
* Ward, N., & Tsukahara, W. (2000). Prosodic features which cue back-channel responses in English and
  Japanese. *Journal of Pragmatics*, 32(8), 1177–1207.
