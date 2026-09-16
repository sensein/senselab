#!/usr/bin/env nextflow
/*
 * ============================================================================================
 * The audio triage DAG.
 *
 *   ADMIT -> PREPROCESS -> TAXONOMY -> AIRWAY -> SPEECH -> { REDACT | VOICE } -> VERDICT
 *                                                            (concurrent)
 *
 * The braces in the brief read `{ AIRWAY, SPEECH -> REDACT, VOICE }`. That fan-out is not
 * realisable from the node documents as written, and the reason is the store rather than an
 * oversight. See `STORE_EDGES` below and the README's "Design tensions".
 *
 * The store is an append-only log of single-author, content-addressed segments. `STORE_EDGES`
 * declares, per node, whose segments are VISIBLE to it. A node reads whatever it likes inside its
 * visibility set and records what it read. That is the whole reconciliation: element-level ports
 * are gone, segment-level visibility remains, and because segments are disjoint by author the
 * merged view is order-independent, so each node is still a pure function of its inputs.
 * ============================================================================================
 */

nextflow.enable.dsl = 2

include { ADMIT      } from './modules/admit.nf'
include { PREPROCESS } from './modules/preprocess.nf'
include { TAXONOMY   } from './modules/taxonomy.nf'
include { AIRWAY     } from './modules/airway.nf'
include { SPEECH     } from './modules/speech.nf'
include { REDACT     } from './modules/redact.nf'
include { VOICE      } from './modules/voice.nf'
include { STORE_VIEW } from './modules/store_view.nf'
include { VERDICT    } from './modules/verdict.nf'

/*
 * --------------------------------------------------------------------------------------------
 * STORE_EDGES — whose segments each node may see, and the sentence in the design that put the
 * edge there. This table is the DAG. It is derived, line by line, from the "what it reads in
 * practice" tables that `store.md` says replaced the port contracts.
 *
 *   PREPROCESS  <- (nothing)             the store is empty when it runs
 *   TAXONOMY    <- PREPROCESS            taxonomy.md: `taxonomy(store)`
 *   AIRWAY      <- PREPROCESS, TAXONOMY  branch-airway.md: spans, silence, ASR words, spectrograms
 *   SPEECH      <- ... , AIRWAY          branch-speech.md step 4: "a segment inside the interval
 *                                        that overlaps an airway_spans entry is withdrawn"
 *   VOICE       <- ... , AIRWAY, SPEECH  branch-voice.md step 1: the residual subtracts
 *                                        airway-labelled spans AND speech spans
 *   REDACT      <- ... , SPEECH          redact.md: acts on SPEECH's PII marking
 *
 * The two edges into SPEECH and VOICE are what serialise the branches. They are not optional in
 * any way that a scheduler can exploit: `branch-speech.md` writes "AIRWAY, if present", and
 * running SPEECH concurrently with AIRWAY would make its step-4 withdrawals depend on which
 * process happened to finish first. That is precisely "execution order significant in a way the
 * DAG does not declare", so the edge is drawn.
 * --------------------------------------------------------------------------------------------
 */

// ============================================================================================
// Entry workflow
//
// `--node <NAME>` runs one node standalone; omitted, it runs the whole graph. Nextflow 26's strict
// parser dropped `-entry`, so the dispatch is a parameter rather than a set of named entry
// workflows — the capability is the same, the flag is different.
// ============================================================================================

workflow {

    if( params.node ) {
        singleNode()
        return
    }

    if( params.validate_params && !workflow.stubRun )
        validateParams()
    else
        validateLayout()          // the publish-root separation is checked even in stub mode

    ch_audio = channel
        .fromPath(params.input, checkIfExists: true)
        .map { f -> [ [ id: safeId(f.baseName) ], f ] }

    ch_hints = params.hints ? file(params.hints, checkIfExists: true) : file("${projectDir}/assets/no-hints.json")

    TRIAGE( ch_audio, ch_hints )
}

// ============================================================================================
// The graph
// ============================================================================================

workflow TRIAGE {

    take:
    ch_audio          // [ meta, path(recording) ]
    hints             // path(hints.json), one file for the run

    main:

    ch_versions = channel.empty()

    // ---- ADMIT -----------------------------------------------------------------------------
    // ADMIT writes NO elements: its product is a decoded signal, and `store.md`'s element kinds
    // (span, word, speaker, interval, measurement) contain nothing it could be. So the store
    // begins at PREPROCESS, and ADMIT contributes a verdict only.
    ADMIT( ch_audio, nodeConfig('admit') )
    ch_versions = ch_versions.mix(ADMIT.out.versions)

    // A `fail` from ADMIT emits no `audio`, so nothing downstream runs. This is the one place in
    // the graph where an outcome controls execution, and `verdict.md` row 1 requires it: "ADMIT
    // failed -> fail, nothing ran". Every OTHER branch fail leaves its branch running, because
    // `verdict.md`'s contradiction rows need a branch outcome even where TAXONOMY said absent.
    ch_admitted = ADMIT.out.audio

    // ---- PREPROCESS ------------------------------------------------------------------------
    PREPROCESS( ch_admitted, nodeConfig('preprocess') )
    ch_versions = ch_versions.mix(PREPROCESS.out.versions)

    // The heavy derivatives (spectrograms, gammatone, ASR output) travel as a directory beside
    // the segment. The segment carries their content hashes; the directory carries the arrays.
    ch_pp = PREPROCESS.out.segment.join(PREPROCESS.out.derivatives)   // [meta, seg, derivs]

    // ---- TAXONOMY --------------------------------------------------------------------------
    ch_tax_in = ch_admitted.join(ch_pp).map { meta, audio, seg, derivs ->
        [ meta, audio, derivs, [ seg ] ]
    }
    TAXONOMY( ch_tax_in, hints, nodeConfig('taxonomy') )
    ch_versions = ch_versions.mix(TAXONOMY.out.versions)

    // ---- AIRWAY ----------------------------------------------------------------------------
    // TAXONOMY does NOT gate this. See the ambiguity list in the README: `verdict.md`'s
    // "absent + pass -> flag" row is unreachable unless every branch runs unconditionally.
    ch_airway_in = ch_admitted.join(ch_pp).join(TAXONOMY.out.segment)
        .map { meta, audio, seg, derivs, tax -> [ meta, audio, derivs, [ seg, tax ] ] }
    AIRWAY( ch_airway_in, hints, nodeConfig('airway') )
    ch_versions = ch_versions.mix(AIRWAY.out.versions)

    // ---- SPEECH ----------------------------------------------------------------------------
    ch_speech_in = ch_admitted.join(ch_pp).join(TAXONOMY.out.segment).join(AIRWAY.out.segment)
        .map { meta, audio, seg, derivs, tax, air -> [ meta, audio, derivs, [ seg, tax, air ] ] }
    SPEECH( ch_speech_in, hints, nodeConfig('speech') )
    ch_versions = ch_versions.mix(SPEECH.out.versions)

    // ---- REDACT and VOICE run concurrently -------------------------------------------------
    // This is the only genuine intra-recording concurrency the node documents permit.
    ch_after_speech = ch_admitted.join(ch_pp).join(TAXONOMY.out.segment)
        .join(AIRWAY.out.segment).join(SPEECH.out.segment)
        .map { meta, audio, seg, derivs, tax, air, spe -> [ meta, audio, derivs, [ seg, tax, air, spe ] ] }

    VOICE( ch_after_speech, hints, nodeConfig('voice') )
    ch_versions = ch_versions.mix(VOICE.out.versions)

    // REDACT is joined on SPEECH's transcript marker, which is a PRODUCT rather than an outcome:
    // it exists iff at least one recognizer returned a word. A recording with no speech therefore
    // never reaches REDACT, and `verdict.md` gives release = `not_assessed` — which it is careful
    // to say is NOT `releasable`, because the audio was never examined for content the transcript
    // could not carry. Reading SPEECH's `outcome` to decide this instead would put a verdict in
    // the control flow, which is the thing that breaks resumability.
    ch_redact_in = ch_after_speech.join(SPEECH.out.marker)
        .map { meta, audio, derivs, segs, _marker -> [ meta, audio, derivs, segs ] }

    ch_redact_segment = channel.empty()
    ch_redact_verdict = channel.empty()
    if( params.redact.enabled ) {
        REDACT( ch_redact_in, nodeConfig('redact') )
        ch_redact_segment = REDACT.out.segment
        ch_redact_verdict = REDACT.out.verdict
        ch_versions       = ch_versions.mix(REDACT.out.versions)
    }

    // ---- The store, gathered ---------------------------------------------------------------
    // Order-independent by construction: segments are disjoint by author, so this `mix` is a
    // join over a grow-only set and `groupTuple` may deliver them in any order.
    ch_segments = PREPROCESS.out.segment
        .mix( TAXONOMY.out.segment, AIRWAY.out.segment, SPEECH.out.segment, VOICE.out.segment, ch_redact_segment )
        .groupTuple(by: 0)

    ch_verdicts = ADMIT.out.verdict
        .mix( PREPROCESS.out.verdict, TAXONOMY.out.verdict, AIRWAY.out.verdict,
              SPEECH.out.verdict, VOICE.out.verdict, ch_redact_verdict )
        .groupTuple(by: 0)

    // ---- STORE_VIEW: the materialised fold -------------------------------------------------
    STORE_VIEW( ch_segments, nodeConfig('store_view') )
    ch_versions = ch_versions.mix(STORE_VIEW.out.versions)

    // ---- VERDICT: the last fold ------------------------------------------------------------
    // `remainder: true` is load-bearing. When ADMIT fails there are no segments at all, and the
    // recording must still reach VERDICT so that triage row 1 can fire.
    ch_verdict_in = ch_verdicts.join(ch_segments, remainder: true)
        .map { meta, verdicts, segments -> [ meta, verdicts ?: [], segments ?: [] ] }

    VERDICT( ch_verdict_in, nodeConfig('verdict') )
    ch_versions = ch_versions.mix(VERDICT.out.versions)

    emit:
    file_verdict = VERDICT.out.verdict
    store        = STORE_VIEW.out.view
    versions     = ch_versions
}

// ============================================================================================
// Single-node execution.
//
// Every process is independently runnable. Hand it a recording, a directory of the segments its
// visibility set would have contained, and (where it needs them) PREPROCESS's derivatives. Nothing
// about a node's behaviour depends on having been reached through the graph, because its whole
// input is files — which is the practical payoff of the store being a set of segments on disk
// rather than an object threaded through channels.
//
//   nextflow run . --node AIRWAY \
//       --input rec.wav \
//       --store_in results/store/rec/ \
//       --derivatives path/to/derivatives
//
// The one thing that does NOT survive standalone execution is the visibility discipline: you are
// handing the node a directory and it will read whatever is in it. Give it the segments the graph
// would have given it, not the whole store, or you are testing a different node.
// ============================================================================================

workflow singleNode {

    main:
    validateLayout()
    def node = params.node.toUpperCase()

    if( node == 'ADMIT' ) {
        ADMIT( soloAudio(), nodeConfig('admit') )
    }
    else if( node == 'PREPROCESS' ) {
        PREPROCESS( soloAudio(), nodeConfig('preprocess') )
    }
    else if( node == 'TAXONOMY' ) {
        TAXONOMY( soloStore(node), soloHints(), nodeConfig('taxonomy') )
    }
    else if( node == 'AIRWAY' ) {
        AIRWAY( soloStore(node), soloHints(), nodeConfig('airway') )
    }
    else if( node == 'SPEECH' ) {
        SPEECH( soloStore(node), soloHints(), nodeConfig('speech') )
    }
    else if( node == 'VOICE' ) {
        VOICE( soloStore(node), soloHints(), nodeConfig('voice') )
    }
    else if( node == 'REDACT' ) {
        REDACT( soloStore(node), nodeConfig('redact') )
    }
    else if( node == 'STORE_VIEW' ) {
        STORE_VIEW( soloSegments(node), nodeConfig('store_view') )
    }
    else if( node == 'VERDICT' ) {
        VERDICT( soloVerdicts(), nodeConfig('verdict') )
    }
    else {
        error "--node ${params.node} is not a node. One of: ADMIT PREPROCESS TAXONOMY AIRWAY SPEECH VOICE REDACT STORE_VIEW VERDICT"
    }
}

// ============================================================================================
// Helpers
// ============================================================================================

/*
 * Parameter ports survive; data ports do not.
 *
 * `archive/ports.md` §1.3 defined two kinds of input port: a DATA port wired to another task's
 * output, and a PARAMETER port wired to one key of the versioned config. `store.md` retires the
 * first and says nothing about the second, so the second stays. `nodeConfig` is that discipline:
 * each node receives its own slice, not the whole params object.
 *
 * This is also the only lever left against over-invalidation under `-resume`. A node's task hash
 * covers its config slice, so tightening a slice tightens the cache key.
 */
def nodeConfig(String node) {
    def undecided = params.undecided
    def models    = params.models
    def common    = [ node: node, device: params.device, pipeline_version: workflow.manifest.version ]
    def empty     = [ decided: [:], undecided: [:], models: [:] ]

    if( node == 'admit' )
        return common + empty
    if( node == 'store_view' )
        return common + empty
    if( node == 'verdict' )
        return common + empty
    if( node == 'preprocess' )
        return common + [
            decided  : params.preprocess,
            undecided: [:],
            models   : models.subMap(['crisperwhisper', 'asr_second', 'alignment', 'squim', 'yamnet'])
        ]
    if( node == 'taxonomy' )
        return common + [
            decided  : params.taxonomy,
            undecided: [ min_families: undecided.taxonomy_min_families ],
            models   : models.subMap(['yamnet', 'ast', 'crisperwhisper', 'hear'])
        ]
    if( node == 'airway' )
        return common + [
            decided  : params.airway,
            undecided: [:],
            models   : models.subMap(['hear', 'yamnet'])
        ]
    if( node == 'speech' )
        return common + [
            decided  : params.speech,
            undecided: [ word_gap_ms: undecided.speech_word_gap_ms, squim: undecided.speech_squim_thresholds ],
            models   : models.subMap(['diarizer', 'diarizer_second', 'separation', 'squim', 'yamnet', 'pii'])
        ]
    if( node == 'voice' )
        return common + [
            decided  : params.voice,
            undecided: [ gate: undecided.voice_gate ],
            models   : [:]                              // branch-voice.md: it measures, it does not classify
        ]
    if( node == 'redact' )
        return common + [
            decided  : params.redact,
            undecided: [ padding_ms: undecided.redact_padding_ms ],
            models   : models.subMap(['crisperwhisper', 'alignment', 'pii'])
        ]
    error "nodeConfig: unknown node '${node}'"
}

/*
 * The publish-root check. Checked on every entry point including stub, because a stub run that
 * writes a release directory inside the store directory has taught the operator the wrong shape.
 */
def validateLayout() {
    def store   = file(params.store_dir).toAbsolutePath().normalize().toString()
    def release = file(params.release_dir).toAbsolutePath().normalize().toString()

    if( store == release )
        error "store_dir and release_dir are the same directory (${store}). The store is never releasable."
    if( release.startsWith(store + '/') )
        error "release_dir (${release}) is inside store_dir (${store}). Publishing a releasable artifact under a sensitive root loses the distinction the moment someone copies the parent."
    if( store.startsWith(release + '/') )
        error "store_dir (${store}) is inside release_dir (${release}). That publishes the unredacted transcript into the release tree."
}

def validateParams() {
    validateLayout()

    if( !params.input )
        error "--input is required"

    // A model is loaded by resolved commit, never by ref. `revision` must be 40 hex characters.
    // A node whose model is not needed is not checked; a node whose model IS needed and whose
    // revision is a ref fails here rather than recording provenance that is confidently wrong.
    def needed = requiredModels()
    def bad = []
    needed.each { role ->
        def m = params.models[role]
        if( !m?.id )
            bad << "models.${role}.id is null — no design document names this model, so you must supply it"
        else if( !m.revision )
            bad << "models.${role}.revision is null — resolve the ref to a commit and pass the 40-hex sha"
        else if( !(m.revision ==~ /^[0-9a-f]{40}$/) )
            bad << "models.${role}.revision = '${m.revision}' is not a 40-hex commit. A ref binds nothing: it may have moved since you read it"
    }
    if( bad )
        error "Provenance is not optional (store.md).\n  - " + bad.join("\n  - ")

    // REDACT's margin. There is no interval to evaluate over, so there is nothing to do but refuse.
    if( params.redact.enabled && params.undecided.redact_padding_ms.value == null )
        error """\
            redact.enabled = true but undecided.redact_padding_ms.value is null.

            REDACT pads every redacted extent outward by a margin that must exceed the WORST
            measured alignment edge error. That distribution has not been measured
            (benchmarks/open.md). There is no admissible interval here, only an unquantified
            bound, so no value can be evaluated over a range and none is defaulted.

            Either supply --undecided.redact_padding_ms.value <ms> with your own derivation, or
            leave redact.enabled = false and accept release = not_assessed.
            """.stripIndent()

    log.info """
        ------------------------------------------------------------------
        triage  ${workflow.manifest.version}
          input        : ${params.input}
          store_dir    : ${params.store_dir}     (SENSITIVE — holds PII after SPEECH)
          release_dir  : ${params.release_dir}   (releasable — REDACT artifacts only)
          work_dir     : ${workflow.workDir}     (SENSITIVE — holds every intermediate)
          REDACT       : ${params.redact.enabled ? 'enabled' : 'disabled -> release = not_assessed'}
        ------------------------------------------------------------------
        """.stripIndent()
}

/*
 * Which model roles this run actually needs. Roles the design leaves optional (a second
 * diarizer, a second ASR, a target embedding) are only required once configured.
 */
def requiredModels() {
    def needed = ['yamnet', 'ast', 'hear', 'crisperwhisper', 'alignment', 'squim', 'diarizer', 'separation', 'pii']
    ['asr_second', 'diarizer_second'].each { role ->
        if( params.models[role]?.id ) needed << role
    }
    return needed
}

/* --- helpers for single-node execution --- */

def soloAudio() {
    return channel.fromPath(params.input, checkIfExists: true)
        .map { f -> [ [ id: safeId(f.baseName) ], f ] }
}

/* [ meta, recording, derivatives, [segments...] ] */
def soloStore(String node) {
    if( !params.store_in )
        error "--node ${node} needs --store_in <dir of segment.*.jsonl>. A node reads the store; it is not handed products."
    def derivs = file(params.derivatives ?: "${projectDir}/assets/empty-derivatives")
    return channel.fromPath(params.input, checkIfExists: true)
        .map { f -> [ [ id: safeId(f.baseName) ], f, derivs, filesIn(params.store_in, 'segment.*.jsonl') ] }
}

def soloSegments(String node) {
    if( !params.store_in )
        error "--node ${node} needs --store_in <dir of segment.*.jsonl>"
    return channel.fromPath(params.input, checkIfExists: true)
        .map { f -> [ [ id: safeId(f.baseName) ], filesIn(params.store_in, 'segment.*.jsonl') ] }
}

def soloVerdicts() {
    if( !params.store_in )
        error "--node VERDICT needs --store_in <dir holding verdict.*.json and segment.*.jsonl>"
    return channel.fromPath(params.input, checkIfExists: true)
        .map { f -> [ [ id: safeId(f.baseName) ],
                      filesIn(params.store_in, 'verdict.*.json'),
                      filesIn(params.store_in, 'segment.*.jsonl') ] }
}

def soloHints() {
    return params.hints ? file(params.hints, checkIfExists: true) : file("${projectDir}/assets/no-hints.json")
}

def filesIn(dir, String pattern) {
    if( !dir ) return []
    def rx = globToRegex(pattern)
    return file(dir).list().findAll { name -> name ==~ rx }.collect { name -> file("${dir}/${name}") }
}

def globToRegex(String glob) {
    return glob.replace('.', '\\.').replace('*', '.*')
}

/* Recording ids reach filesystem paths and a store key. Keep them boring. */
def safeId(String raw) {
    def clean = raw.replaceAll(/[^A-Za-z0-9._-]/, '_')
    if( !clean ) error "recording basename '${raw}' reduces to an empty id"
    return clean
}
