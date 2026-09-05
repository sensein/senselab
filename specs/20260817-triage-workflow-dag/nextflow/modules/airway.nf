/*
 * AIRWAY branch.
 *
 * `branch-airway.md`: it proposes no elements of its own. It `label`s, `confirm`s and `contest`s the
 * `span` elements PREPROCESS wrote. HeAR classifies the whole span in a 2 s buffer; YAMNet
 * confirms or contests from its own native 0.96 s windows by coverage; an ASR word inside
 * [first labelled span start, last labelled span end] flags the file.
 *
 * A `contest` does not resolve. Both assertions land in the segment and the outcome is `flag`.
 * That is `store.md`'s rule and it is why the segment is append-only rather than a table of
 * current labels: there is no column for "both instruments, disagreeing".
 *
 * The figure is an artifact beside the store, not an element. It carries element ids for
 * traceability, which is correct here and disqualifying in a released artifact — so it is
 * published under `store_dir` and REDACT is the only node that may write to `release_dir`.
 *
 * Visibility: PREPROCESS + TAXONOMY. This branch reads no other branch, which is what lets it run
 * first rather than last.
 */

process AIRWAY {

    tag   "${meta.id}"
    label 'triage_model'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*',
               saveAs: { fn -> fn.substring(fn.lastIndexOf('/') + 1) }

    input:
    tuple val(meta), path(audio), path(derivatives, stageAs: 'derivatives'), path(store_in, stageAs: 'store_in/*')
    path  hints
    val   node_config

    output:
    tuple val(meta), path("store/segment.airway.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.airway.json"),    emit: verdict
    tuple val(meta), path("store/figure.airway.*"),        emit: figure, optional: true
    path  "versions.yml",                                  emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def models = Triage.modelsArg(node_config.models)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store store_in
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node AIRWAY \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --derivatives derivatives \\
        --store-in store_in \\
        --hints '${hints}' \\
        --config node-config.json \\
        ${models} \\
        --out store \\
        --figure store/figure.airway.png \\
        ${replay}

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store store_in
    printf '%s' ${cfg} > node-config.json

    triage-node --node AIRWAY --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' --derivatives derivatives \\
        --store-in store_in --hints '${hints}' --config node-config.json --out store \\
        --figure store/figure.airway.png

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
