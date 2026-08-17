# The triage workflow, as a graph

**This page is self-contained.** It is the picture of the proposed audio-triage workflow: what it
asks about a recording, in what order, what evidence each answer needs, and where it loops. Nothing
here depends on reading the prose in `design.md`; the prose exists to justify these diagrams, not to
explain them. Port tables for every task are in `ports.md`; the executable-shaped version is
`workflow.nf`.

Every box is a **task**. A task has named input ports and named output ports and nothing else — it
cannot read anything that is not wired to an input port. A task is either a plain function call or a
sub-workflow; from the outside these are indistinguishable, which is why the overview can show a
sub-workflow as one box and a later figure can open it up.

Edge labels are **port names**, so an edge reads `producer output → consumer input`. Where an edge
carries a parameter rather than a product, its label starts with `cfg.` and names the key in
`src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`.

## Legend

| Shape | Kind | What it means |
| --- | --- | --- |
| `[[ double square ]]` | model inference | Loads a model and runs it. Cached, costs GPU or a subprocess venv. |
| `[ square ]` | pure computation | Deterministic function of its declared inputs. No model, no cache. |
| `{ diamond }` | decision gate | Turns measurements into a published `Estimate` under a named config threshold. Also the only thing that may filter a stream. |
| `([ stadium ])` | terminal product | Reaches the caller. Every one is an `Estimate` plus, where relevant, spans. |
| `[/ parallelogram /]` | sub-workflow | A task whose body is another graph. Same port contract as any other task. |

Colour repeats the shape so the figures stay readable in greyscale and for colour-blind readers.

---

## Figure 1 — Overview: the questions, in dependency order

Read top to bottom. Each box answers one question about the recording; the edges are the evidence
that has to exist before the next question is answerable. The boxes here are sub-workflows; the
figures after this one open each of them.

```mermaid
flowchart TD
    IN(["Recording, and optionally: the task the participant was asked to do,<br/>a known sample of the target voice"])
    CFG[/"One versioned config<br/>model ids, grids, thresholds, each with a written derivation"/]

    W1[/"ADMIT<br/>Is there any usable signal at all"/]
    W2[/"TAXONOMY<br/>What kinds of sound are in here:<br/>lexical speech, non-lexical voice, non-vocal sound, silence"/]
    W3[/"SPEECH CONTENT<br/>What was said, does it match the asked task,<br/>is there personal information in it"/]
    W4[/"VOICE IDENTITY, on the raw audio only<br/>How many voices, and is any of them not the target"/]
    W5[/"QUALITY<br/>Is this recording good enough to measure"/]
    W6[/"TRIM<br/>Which regions should be cut, and why each"/]
    W7[/"DECIDE<br/>Does a human need to look at this, and why"/]
    W8[/"REFINE<br/>Is any answer still undecided, and would more compute change it"/]

    DEAD(["Answer: unusable file.<br/>Flagged, with the reason, and nothing else claimed"])
    OUT(["Nine answers, each with its evidence count,<br/>its prior, and the population it was validated on"])

    IN -->|"audio_file"| W1
    CFG -.->|"cfg parameters, per task"| W1
    CFG -.->|"cfg parameters, per task"| W2
    CFG -.->|"cfg parameters, per task"| W3
    CFG -.->|"cfg parameters, per task"| W4
    CFG -.->|"cfg parameters, per task"| W5

    W1 -->|"no usable signal"| DEAD
    W1 -->|"audio_raw, audio_enhanced,<br/>level_track, band_floor"| W2

    W2 -->|"vocal_spans, taxonomy_track"| W3
    W2 -->|"vocal_spans, taxonomy_track"| W4
    W2 -->|"taxonomy_track, content"| W5
    W2 -->|"content"| W7

    W3 -->|"transcript, word_times"| W4
    W3 -->|"transcript, task_match, pii, pii_spans"| W7
    W3 -->|"pii_spans"| W6

    W4 -->|"speaker_count, off_target, off_target_spans"| W7
    W4 -->|"off_target_spans"| W6

    W5 -->|"quality, defect_spans"| W7
    W5 -->|"defect_spans"| W6

    W6 -->|"trim_regions"| W7

    W7 -->|"ledger"| W8
    W8 -.->|"round k plus 1: same graph,<br/>narrowed to the undecided regions"| W2
    W8 -->|"stop_reason"| OUT
    W7 -->|"review_flag, reasons, and the nine answers"| OUT

    classDef sub fill:#ede9fe,stroke:#5b21b6,stroke-width:2px,color:#1e1b4b
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    class W1,W2,W3,W4,W5,W6,W7,W8,CFG sub
    class IN,OUT,DEAD prod
```

**What the dashed edge from REFINE means.** It is not a cycle in the data. Round *k*'s products are
distinct, immutable values from round *k+1*'s; the ledger that crosses the boundary is an input
product to the next round and is never written by it. Figure 8 shows the loop and its exit criteria.

**Why VOICE IDENTITY reads the raw audio only.** Enhancement exists to suppress background voices,
which is exactly the evidence an off-target check is looking for. The rule is structural here: the
enhanced variant has no wire into that sub-workflow, so it cannot be used by accident.

---

## Figure 2 — ADMIT: is there anything here, and which variants exist

```mermaid
flowchart TD
    F(["audio_file"])
    C1{{"cfg.device"}}
    C2{{"cfg.models.enhancement"}}
    C3{{"cfg.quality.floor_percentile"}}
    C4{{"cfg.triage.speech_threshold, cfg.triage.min_speech_s"}}

    T01[["decode_audio<br/>read, downmix to mono, resample"]]
    T02["level_and_floor<br/>loudness, true peak, clipped fraction,<br/>per-band noise floor with bias correction"]
    T03{"signal_gate<br/>Is there measurable acoustic content"}
    T04[["enhance_audio<br/>produces a second variant of the same recording"]]

    P1(["audio_raw"])
    P2(["audio_enhanced"])
    P3(["level_track, band_floor, clip_track"])
    P4(["signal_present, an Estimate"])
    DEAD(["Stream ends here.<br/>Consumers get no input, so they do not run"])

    F -->|"audio_file"| T01
    C1 -.-> T01
    T01 -->|"audio_raw"| T02
    C3 -.-> T02
    T02 -->|"level_track, band_floor"| T03
    C4 -.-> T03
    T03 -->|"not present"| DEAD
    T03 -->|"present"| P1
    T03 -->|"present"| P3
    T03 -->|"present"| P4
    P1 -->|"audio_raw"| T04
    C2 -.-> T04
    T04 -->|"audio_enhanced"| P2

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    class T01,T04 inf
    class T02 pure
    class T03 gate
    class F,P1,P2,P3,P4,DEAD prod
    class C1,C2,C3,C4 par
```

**The gate is the skip mechanism.** There is no `skip_stages` flag anywhere in this design. A gate
that decides the file is unusable emits no product on its downstream port, so every consumer has an
empty input and does not run. That is the only way a stage is ever skipped.

---

## Figure 3 — TAXONOMY: what kinds of sound are in here

This is the root answer of the whole workflow, and the part that today's pipeline does not have. It
replaces a binary "was there speech" with a four-way statement, because a cough, a cry or a groan is
neither speech nor background noise, and forcing it into either answer is wrong in a way that
propagates into trimming and into speaker attribution.

```mermaid
flowchart TD
    A(["audio_raw"])
    L(["level_track, band_floor"])
    C1{{"cfg.grid.win_length, cfg.grid.hop_length"}}
    C2{{"cfg.scene.top_k"}}
    C3{{"cfg.linking.frame_speech_threshold,<br/>cfg.linking.label_mass_threshold,<br/>cfg.linking.speech_excess_db"}}

    T10[["speech_frame_posterior<br/>frame-level probability of voice, at the model frame rate"]]
    T11[["sound_event_posterior<br/>frame-level scores over the sound-event ontology"]]
    T12[["voicing_track<br/>periodicity, harmonicity, phoneme-vs-silence fraction"]]

    T13["taxonomy_fold<br/>one distribution per window over four classes:<br/>lexical speech, non-lexical voice, non-vocal sound, silence"]
    T14{"content_gate<br/>Which classes are present, and how sure are we"}

    P1(["taxonomy_track, a per-window table"])
    P2(["vocal_spans, covering both lexical and non-lexical voice"])
    P3(["event_posterior, kept for the off-target check"])
    P4(["content, an Estimate per class"])

    A --> T10
    A --> T11
    A --> T12
    C2 -.-> T11
    T10 -->|"speech_posterior"| T13
    T11 -->|"event_posterior"| T13
    T12 -->|"voicing_track"| T13
    L -->|"level_track, band_floor"| T13
    C1 -.-> T13
    C3 -.-> T13
    T13 -->|"taxonomy_track"| T14
    T13 -->|"taxonomy_track"| P1
    T13 -->|"vocal_spans"| P2
    T11 -->|"event_posterior"| P3
    T14 -->|"content"| P4

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    class T10,T11,T12 inf
    class T13 pure
    class T14 gate
    class A,L,P1,P2,P3,P4 prod
    class C1,C2,C3 par
```

**`vocal_spans` is the product that everything about voices hangs on**, and it is word-independent
by construction: it comes from frame posteriors and periodicity, never from whether a recognizer
produced text. That is what makes it usable for a cry, and it is the reason the current pipeline's
"no words here, so clear the speaker evidence" rule has no equivalent in this design.

---

## Figure 4 — SPEECH CONTENT: what was said, and what is in it

```mermaid
flowchart TD
    A(["audio_raw"])
    B(["audio_enhanced"])
    V(["vocal_spans"])
    T(["taxonomy_track"])
    H(["hints.expected_speech, present only when the caller supplied it"])
    C1{{"cfg.models.asr, a list of three recognizers"}}
    C2{{"cfg.alignment.aligner, cfg.alignment.qwen_model, cfg.alignment.language"}}
    C3{{"cfg.linking.asr_slot_overlap, cfg.linking.asr_slot_mid_tol_s"}}
    C4{{"no config key exists — see the unwired-ports note"}}

    T20[["transcribe<br/>one call per recognizer, over the whole usable audio"]]
    T21[["align_words<br/>one aligner, so boundary differences are attributable"]]
    T22["fuse_words<br/>group by sequence alignment, grade phonemically,<br/>one confidence per word and per edge"]
    T23{"transcript_gate<br/>Is the transcript trustworthy, and where is it not"}
    T24{"task_match_gate<br/>Does the content match what was asked"}
    T25[["pii_scan<br/>one call per detector: rules, Presidio, GLiNER"]]
    T26{"pii_gate<br/>Is personal information present, and where"}

    P1(["transcript, words with confidence and edge confidences"])
    P2(["word_times"])
    P3(["transcript_confidence"])
    P4(["task_match, absent when no expected task was supplied"])
    P5(["pii, pii_spans"])

    A --> T20
    B --> T20
    C1 -.-> T20
    T20 -->|"hypotheses per model"| T21
    A --> T21
    C2 -.-> T21
    T21 -->|"word_times per model"| T22
    C3 -.-> T22
    T22 -->|"transcript"| T23
    V -->|"vocal_spans: which spans should have produced words"| T23
    T22 -->|"transcript"| P1
    T22 -->|"word_times"| P2
    T23 -->|"transcript_confidence"| P3

    T22 -->|"transcript"| T24
    T -->|"taxonomy_track"| T24
    H -->|"expected_speech"| T24
    T24 -->|"task_match"| P4

    T22 -->|"transcript"| T25
    C4 -.-> T25
    T25 -->|"pii_candidates per detector"| T26
    P2 -->|"word_times, to place a span in time"| T26
    T26 -->|"pii, pii_spans"| P5

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    classDef bad fill:#fecaca,stroke:#7f1d1d,stroke-width:3px,stroke-dasharray:6 4,color:#1c0a0a
    class T20,T21,T25 inf
    class T22 pure
    class T23,T24,T26 gate
    class A,B,V,T,H,P1,P2,P3,P4,P5 prod
    class C1,C2,C3 par
    class C4 bad
```

**Two things to notice.**

`task_match` has no default. When the caller supplies no expected task, the `expected_speech` port
has no producer, `task_match_gate` does not run, and the output is *absent*. It never becomes a
value meaning "unknown", because a consumer cannot tell a fabricated 0.5 from a measured one.

`transcript_gate` is where `vocal_spans` earns its place: a span the taxonomy calls voice and the
recognizers left empty is a *measured disagreement*, not a silence. Today that same situation is
read as "nothing was said here" and used to erase evidence.

---

## Figure 5a — VOICE IDENTITY as a single task, with its ports

The point of this figure is that a sub-workflow is a task. From the outside, `VOICE_IDENTITY` has
ports and nothing else; the parent graph neither knows nor cares that its body is another graph.

```mermaid
flowchart LR
    I1(["audio_raw"]) --> W
    I2(["vocal_spans"]) --> W
    I3(["taxonomy_track"]) --> W
    I4(["event_posterior"]) --> W
    I5(["transcript, optional"]) --> W
    I6(["hints.target_voice, optional"]) --> W
    P1{{"cfg.models.diarization"}} -.-> W
    P2{{"cfg.models.embeddings"}} -.-> W
    P3{{"cfg.embeddings.window_s, cfg.embeddings.hop_s"}} -.-> W
    P4{{"cfg.speaker.same_floor, cfg.speaker.diff_floor,<br/>cfg.speaker.cluster_cosine_threshold,<br/>cfg.speaker.clustering_algorithm"}} -.-> W

    W[/"VOICE IDENTITY"/]

    W --> O1(["speaker_count"])
    W --> O2(["off_target"])
    W --> O3(["off_target_spans"])
    W --> O4(["cluster_structure"])
    W --> O5(["overlap_track"])

    classDef sub fill:#ede9fe,stroke:#5b21b6,stroke-width:3px,color:#1e1b4b
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    class W sub
    class I1,I2,I3,I4,I5,I6,O1,O2,O3,O4,O5 prod
    class P1,P2,P3,P4 par
```

## Figure 5b — VOICE IDENTITY expanded

Same ports, opened up. The dangling names at the top and bottom are the ports from Figure 5a.

```mermaid
flowchart TD
    A(["audio_raw"])
    V(["vocal_spans"])
    T(["taxonomy_track"])
    E(["event_posterior"])
    TR(["transcript, optional"])
    HV(["hints.target_voice, optional"])

    T30[["window_embeddings<br/>one call per embedder, windows drawn on vocal_spans"]]
    T31["cluster_windows<br/>candidate speaker counts with a separation score each,<br/>plus the dominant cluster and its share of voiced time"]
    T32[["diarize<br/>one call per diarizer"]]
    T33["harmonize_labels<br/>map every diarizer into one label space,<br/>and derive where they say two voices overlap"]
    T34{"speaker_count_gate<br/>How many distinct voices, with an evidence count"}
    T35["novelty_track<br/>per window, distance from the dominant voice"]
    T36["off_target_fold<br/>combine novelty, overlap, background-voice scores,<br/>and where a reference passage exists, transcript deviation"]
    T37{"off_target_gate<br/>Is a voice other than the majority voice present"}

    O1(["speaker_count"])
    O2(["off_target"])
    O3(["off_target_spans"])
    O4(["cluster_structure"])
    O5(["overlap_track"])

    A --> T30
    V -->|"vocal_spans"| T30
    T30 -->|"window_embeddings"| T31
    T31 -->|"cluster_structure"| T34
    T31 -->|"cluster_structure"| O4
    A --> T32
    T32 -->|"diarization per model"| T33
    T30 -->|"window_embeddings, for centroid matching"| T33
    T33 -->|"harmonized_speakers"| T34
    T33 -->|"overlap_track"| O5
    T34 -->|"speaker_count"| O1

    T31 -->|"cluster_structure: dominant centroid"| T35
    T30 -->|"window_embeddings"| T35
    T35 -->|"novelty_track"| T36
    T33 -->|"overlap_track"| T36
    E -->|"event_posterior: babble, crowd, chatter subtree"| T36
    T -->|"taxonomy_track"| T36
    TR -->|"transcript, for read-passage deviation"| T36
    T36 -->|"off_target_track"| T37
    HV -->|"target_voice, promotes novelty to verification"| T37
    T37 -->|"off_target"| O2
    T37 -->|"off_target_spans"| O3

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    class T30,T32 inf
    class T31,T33,T35,T36 pure
    class T34,T37 gate
    class A,V,T,E,TR,HV,O1,O2,O3,O4,O5 prod
```

**The honest claim this sub-workflow makes.** With no enrolled sample of the target voice, "someone
other than the majority voice is present" is defensible and "that person is not the participant" is
not. So `off_target` is published as the former, and `hints.target_voice` — when a caller has one —
is what upgrades it from novelty detection to verification. The distinction is a wire, not a caveat
in prose.

---

## Figure 6 — QUALITY and TRIM

```mermaid
flowchart TD
    A(["audio_raw"])
    B(["audio_enhanced"])
    L(["level_track, band_floor, clip_track"])
    T(["taxonomy_track"])
    CT(["content"])
    OS(["off_target_spans"])
    PS(["pii_spans"])
    C1{{"cfg.quality.analysis_win_length, cfg.quality.analysis_hop_length"}}
    C2{{"cfg.grid.win_length, cfg.grid.hop_length"}}
    C3{{"cfg.profiles.calibration, cfg.profiles.detection_margin"}}

    T40[["scene_quality_frames<br/>frame signal-to-noise, reverberation, voice activity"]]
    T41["quality_measures<br/>decibels, hertz and proportions on the reporting grid.<br/>Nothing here is a score"]
    T42{"degradation_gate<br/>How degraded is each axis, against anchors<br/>chosen for the content class that was found"}
    T43["defect_spans<br/>clipping, dropout, level excursions"]
    T50["trim_proposal<br/>every candidate region with the reason it is a candidate"]
    T51{"trim_gate<br/>Which regions to propose cutting, each with its own Estimate"}

    O1(["quality, an Estimate per axis"])
    O2(["defect_spans"])
    O3(["trim_regions"])

    A --> T40
    B --> T40
    C1 -.-> T40
    T40 -->|"snr_track, c50_track, bandwidth_track"| T41
    L -->|"level_track, band_floor, clip_track"| T41
    C2 -.-> T41
    T41 -->|"quality_measures"| T42
    CT -->|"content: which anchors apply"| T42
    C3 -.-> T42
    T42 -->|"quality"| O1
    T41 -->|"quality_measures"| T43
    T43 -->|"defect_spans"| O2

    T -->|"taxonomy_track: leading and trailing non-vocal"| T50
    OS -->|"off_target_spans"| T50
    PS -->|"pii_spans"| T50
    T43 -->|"defect_spans"| T50
    T50 -->|"trim_candidates"| T51
    T51 -->|"trim_regions"| O3

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    class T40 inf
    class T41,T43,T50 pure
    class T42,T51 gate
    class A,B,L,T,CT,OS,PS,O1,O2,O3 prod
    class C1,C2,C3 par
```

**Every trim region names its reason and carries its own confidence.** A region proposed because the
taxonomy found no voice there is a different claim from one proposed because a non-target voice was
found there, and a consumer that cannot tell them apart will eventually cut a participant's cough
out of a cough recording.

**The degradation anchors depend on the content class.** What counts as clean for fluent read speech
is not what counts as clean for a breathing task, so `content` is a declared input to that gate
rather than an assumption inside it.

---

## Figure 7 — DECIDE: the flag, and what it is allowed to say

```mermaid
flowchart TD
    E1(["content"])
    E2(["transcript_confidence"])
    E3(["speaker_count"])
    E4(["off_target"])
    E5(["quality"])
    E6(["task_match, may be absent"])
    E7(["pii"])
    E8(["trim_regions"])
    C1{{"cfg.labelstudio.low_threshold, cfg.labelstudio.high_threshold"}}
    C2{{"evidence floor and shrinkage prior per answer,<br/>each with its derivation or marked unfitted"}}

    T60["evidence_ledger<br/>every published answer with its value, its raw statistic,<br/>how many independent sources produced it,<br/>its prior, and the population it was validated on"]
    T61{"review_flag_gate<br/>three arms, each of which names itself when it fires"}

    R1(["Arm 1: an answer is bad enough to matter,<br/>and there was enough evidence to say so"])
    R2(["Arm 2: an answer has too little evidence to adjudicate.<br/>This is also what routes the file to refinement"])
    R3(["Arm 3: the recording contradicts the task it was asked for.<br/>Only evaluated when a task was supplied"])
    OUT(["review_flag, plus ranked reasons.<br/>A flag with no reasons is a bug, not a pass"])
    LED(["ledger, the input to REFINE"])

    E1 --> T60
    E2 --> T60
    E3 --> T60
    E4 --> T60
    E5 --> T60
    E6 --> T60
    E7 --> T60
    E8 --> T60
    C2 -.-> T60
    T60 -->|"ledger"| T61
    T60 -->|"ledger"| LED
    C1 -.-> T61
    T61 --> R1
    T61 --> R2
    T61 --> R3
    R1 --> OUT
    R2 --> OUT
    R3 --> OUT

    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    class T60 pure
    class T61 gate
    class E1,E2,E3,E4,E5,E6,E7,E8,R1,R2,R3,OUT,LED prod
    class C1,C2 par
```

---

## Figure 8 — REFINE: the round loop as conditional re-entry

The loop is not an unrolled copy of the graph. It is one sub-workflow, `REFINE`, whose inputs
include the previous iteration's ledger, and which re-enters the same tasks over a narrower input.
The exit criteria are on the edges.

```mermaid
flowchart TD
    L(["ledger at round k"])
    B(["budget_remaining at round k"])
    HIS(["action_history, every action set already executed"])
    C1{{"cfg.rounds.max_rounds"}}
    C2{{"cfg.rounds.epistemic_tolerance, cfg.rounds.cycle_window"}}

    T70["rank_undecided<br/>which answers are still ambiguous,<br/>and which unused action could add independent evidence to each"]
    T71{"stop_or_continue"}
    T72["narrow_input<br/>cut the audio to the ambiguous regions and re-enter"]
    RE[/"TAXONOMY, SPEECH CONTENT, VOICE IDENTITY, QUALITY, TRIM, DECIDE<br/>the same tasks, over the narrowed input"/]
    LK(["ledger at round k plus 1, a new and distinct value"])
    STOP(["stop_reason, published as an output"])

    L --> T70
    HIS --> T70
    C2 -.-> T70
    T70 -->|"candidate_actions"| T71
    B --> T71
    C1 -.-> T71

    T71 -->|"every answer decisive: DECIDED"| STOP
    T71 -->|"ambiguous, but no unused action would add evidence: IRREDUCIBLE"| STOP
    T71 -->|"planned action set repeats one already executed: OSCILLATING"| STOP
    T71 -->|"budget spent or round cap reached: EXHAUSTED"| STOP
    T71 -->|"ambiguous, an action would add evidence, budget remains"| T72

    T72 -->|"audio_regions, and the ports each task needs"| RE
    RE -->|"fresh measurements over the narrowed regions"| LK
    LK -.->|"becomes ledger at round k for the next iteration"| L

    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef par fill:#f1f5f9,stroke:#475569,color:#0f172a
    classDef sub fill:#ede9fe,stroke:#5b21b6,stroke-width:2px,color:#1e1b4b
    class T70,T72 pure
    class T71 gate
    class L,B,HIS,LK,STOP prod
    class C1,C2 par
    class RE sub
```

**What crosses the boundary, exactly:** the ledger, the remaining budget, and the list of action sets
already executed. Nothing else. No task reads a round index; only `rank_undecided` and
`stop_or_continue` see round state at all, and they see it as ordinary input ports.

**Why this is not a cycle in the data graph.** Products are versioned by round. `ledger@k` is an
input to round *k+1*'s tasks; round *k+1* writes `ledger@k+1`, a different value at a different
name. Nothing ever writes a product that one of its own ancestors read. The cycle exists only in the
graph over *task names*, and task names are not products.

**Stopping is an answer.** `stop_reason` is published, and `IRREDUCIBLE` combined with an ambiguous
answer is the honest terminal state: the tools available cannot decide this file, so a human must.

---

## Figure 9 — Where caching attaches

```mermaid
flowchart LR
    K["Cache key<br/>waveform signature, task name, model id,<br/>resolved commit sha, canonical parameters,<br/>wrapper code version, senselab version, schema version"]
    I1[["decode and enhance"]]
    I2[["frame posteriors, sound events, voicing"]]
    I3[["transcribe, align"]]
    I4[["embeddings, diarize"]]
    I5[["scene quality frames"]]
    I6[["personal-information detectors"]]
    N["Pure computation and gates:<br/>not cached. Cheap, deterministic,<br/>and a second invalidation surface is a liability"]

    K --> I1
    K --> I2
    K --> I3
    K --> I4
    K --> I5
    K --> I6

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef key fill:#e0e7ff,stroke:#3730a3,stroke-width:2px,color:#1e1b4b
    class I1,I2,I3,I4,I5,I6 inf
    class N pure
    class K key
```

A round that narrows the audio to a region hands the inference tasks a **different waveform**, so it
gets a different signature and a different cache entry. That is the intended behaviour and it needs
no new key field — but it does mean the narrowed slice must be materialised as audio and its time
offsets restored on the way out, rather than passed as a span parameter alongside the whole file.

---

## Figure 10 — What today's code wires, and the ports with no producer

For contrast. This is the current pipeline's speaker-attribution path, drawn with the same
conventions. The dashed red box is a port that nothing writes, which is why three decisions
downstream of it have never once fired on a real run.

```mermaid
flowchart TD
    S1[["diarize, several models"]]
    S2[["transcribe, several recognizers"]]
    S3["background mask: regions free of TARGET activity"]
    B1["pass summary, an untyped dictionary<br/>eight keys, read from nine modules at thirty-three sites"]
    M1(["mask counters only. The per-region table goes to a parquet file instead"])
    X(["mask.regions — NO PRODUCER"])
    A1["speaker attribution"]
    D1{"clear this bucket if the mask says target-free"}
    D2{"skip the word gate if the mask reports a voice"}
    D3["target activity voter"]
    OUT(["speaker axis"])

    S1 --> B1
    S2 --> B1
    S3 --> M1
    M1 --> B1
    B1 --> A1
    X -.-> A1
    A1 --> D1
    A1 --> D2
    A1 --> D3
    D1 --> OUT
    D2 --> OUT
    D3 --> OUT

    classDef inf fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0c1a3d
    classDef pure fill:#fef9c3,stroke:#a16207,stroke-width:2px,color:#1c1917
    classDef gate fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#1c0a0a
    classDef prod fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef bad fill:#fecaca,stroke:#7f1d1d,stroke-width:3px,stroke-dasharray:6 4,color:#1c0a0a
    class S1,S2 inf
    class S3,B1,A1,D3 pure
    class D1,D2 gate
    class M1,OUT prod
    class X bad
```

The proposed design has no equivalent of that dashed box, because `vocal_spans` is a declared output
port of a task that measures the audio, and a port with no producer stops the graph instead of
quietly reading an empty list. `design.md` section 7 works through what that means for the three
dead decisions.
