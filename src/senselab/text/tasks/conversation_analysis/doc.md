# Conversation analysis

Turn-taking, backchannels, overlaps and interruptions, disfluencies, rule-based dialogue acts,
engagement/politeness markers and entrainment, computed from a diarized, word-timestamped transcript
of a conversation between two or more speakers. Pure Python: no model is loaded.

```python
from senselab.text.tasks.conversation_analysis import analyze_conversation

words = [
    {"speaker": "A", "start": 0.0, "end": 0.3, "text": "So"},
    {"speaker": "A", "start": 0.35, "end": 0.6, "text": "[UM]"},
    ...
]
result = analyze_conversation(words)
tables = result.to_dataframes()   # words, ipus, turns, overlaps, floor_transfers, disfluencies,
                                  # dialogue_acts, markers, summary (+ entrainment)
per_speaker = result.summary()    # {speaker: {metric: value}}
```

Words may be `ScriptLine`s (or lines whose `chunks` are words) or mappings with `speaker`, `start`,
`end`, `text` and optional `confidence`, in any order. Fillers (`uh`, `um`, `[UH]`, ...) are kept
verbatim everywhere: no step drops or rewrites a word.

## Pipeline

1. **IPUs** — a speaker's words joined across silences shorter than `ipu_gap_s`.
2. **Backchannels** — short IPUs that are lexically backchannel-like or wholly overlapped, and that
   leave the floor with the other speaker.
3. **Turns** — non-backchannel IPUs grouped per floor; an IPU wholly inside another speaker's turn is
   an `embedded` turn that never held the floor.
4. **Floor transfers** — FTO between consecutive floor-holding turns; gap / overlap / no-gap-no-overlap.
5. **Overlaps** — every pair of overlapping IPUs: backchannel, non-competitive (terminal) or
   competitive (interruption, successful or unsuccessful).
6. **Dialogue acts** — turns split at sentence-final punctuation and labelled by
   `label_dialogue_acts(segments, labeller=None)`; pass any `DialogueActLabeller` to replace the rules.
   A turn's act is `primary_act(segments)`: backchannel/agreement never head a turn that also holds a
   question, directive or statement with `substantive_min_content_words` content words.
7. **Disfluencies** — filled pauses (with position, surrounding pauses and what follows), repetitions,
   cut-offs and prolongations.
8. **Markers** — discourse markers, hedges, politeness strategies, pronouns and lexical alignment.
9. **Entrainment** — `compute_entrainment(turn_features)` on any per-turn numeric series,
   `turn_entrainment(turns, features)` on an analysis' floor-holding turns with a `turn_id`-keyed
   feature table, or `analyze_conversation(..., turn_features=df)`.

Thresholds live in `data/defaults.yaml` (each with its derivation) and lexicons in
`data/lexicons.yaml`; override either through `ConversationAnalysisConfig`. Definitions and
references for every metric are in `specs/20261017-120000-conversation-analysis/spec.md`.
