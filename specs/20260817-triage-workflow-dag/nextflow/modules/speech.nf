/*
 * SPEECH branch.
 *
 * `branch-speech.md`: eight steps, no ASR of its own (PREPROCESS ran it), and the point at which
 * THE STORE BECOMES SENSITIVE. From the moment this node writes a `word` element, the store holds
 * an unredacted transcript, and being append-only it stays that way forever. Nothing downstream may
 * treat a clean PII scan as clearance to release audio, and REDACT cannot retroactively clean this
 * segment.
 *
 * Visibility: PREPROCESS + TAXONOMY + AIRWAY. The AIRWAY edge is step 4 — "a segment inside the
 * interval that overlaps an airway_spans entry is withdrawn". `branch-speech.md` marks that read
 * "AIRWAY, if present", which reads like an optional input and is not one for a scheduler: running
 * this node concurrently with AIRWAY makes its withdrawals depend on which process finished first.
 * So the edge is drawn and the two branches are sequential. See the README.
 *
 * Step 8 (SQUIM quality) is described as a parallel branch that blocks nothing, and its outputs are
 * `measurement` elements that do not appear in this node's verdict at all. It could therefore be
 * hoisted into its own process and run concurrently with REDACT and VOICE. It is kept inside for
 * now because splitting it would put a second author on the same recording's quality claims and
 * `store.md` gives no rule for that.
 *
 * The `word_gap_ms` grouping threshold has no measured value. When it is null this node does NOT
 * invent one: it groups by the recognizer's own utterance boundaries and records
 * `span_grouping = "recognizer_native"` in the element's provenance, so a reader can tell which
 * rule produced the span. If the recognizer supplies no utterance boundaries either, the node is
 * unrunnable and says so.
 */

process SPEECH {

    tag   "${meta.id}"
    label 'triage_asr'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*'

    input:
    tuple val(meta), path(audio), path(derivatives, stageAs: 'derivatives'), path(store_in, stageAs: 'store_in/*')
    path  hints
    val   node_config

    output:
    tuple val(meta), path("store/segment.speech.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.speech.json"),    emit: verdict
    tuple val(meta), path("store/figure.speech.*"),        emit: figure, optional: true
    tuple val(meta), path("streams"),                      emit: streams, optional: true
    path  "versions.yml",                                  emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def models = Triage.modelsArg(node_config.models)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store store_in streams
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node SPEECH \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --derivatives derivatives \\
        --store-in store_in \\
        --hints '${hints}' \\
        --config node-config.json \\
        ${models} \\
        --out store \\
        --streams streams \\
        --figure store/figure.speech.png \\
        ${replay}

    # The verdict carries category and extent, never the matched text (branch-speech.md step 7).
    # Cheap, mechanical check that nothing put a `text` key inside a `pii` finding.
    triage-node --node CHECK_VERDICT_NO_PII_TEXT --verdict store/verdict.speech.json

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store store_in streams
    printf '%s' ${cfg} > node-config.json

    triage-node --node SPEECH --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' --derivatives derivatives \\
        --store-in store_in --hints '${hints}' --config node-config.json --out store \\
        --streams streams --figure store/figure.speech.png

    triage-node --node CHECK_VERDICT_NO_PII_TEXT --verdict store/verdict.speech.json

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
