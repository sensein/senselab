/*
 * PREPROCESS — the first writer to the store.
 *
 * `preprocess.md`: no `fail`, no `flag`. A derivative that cannot be computed is simply absent
 * from the store, and a consumer that needs it does not run. So the verdict this node emits always
 * carries outcome `pass`; its informative fields are `derivatives_written` and, when no peak
 * anywhere reaches K above the local floor, `spans: no_contrast` — which AIRWAY reads as one of
 * its `fail` conditions.
 *
 * Two outputs, and the split is deliberate:
 *
 *   store/segment.preprocess.<hash>.jsonl   the elements and assertions. Small. Every downstream
 *                                           node stages this.
 *   derivatives/                            the arrays — spectrograms, gammatone, envelope, ASR
 *                                           output. Large. The segment carries their content
 *                                           hashes, not their contents.
 *
 * The segment is the store; `derivatives/` is the blob half of a content-addressed pair. Splitting
 * them keeps the store readable and diffable, and it is the only thing standing between a reader
 * and a 200 MB JSONL line.
 *
 * `derivatives/` is NOT published by default. It contains the unredacted transcript from the
 * moment ASR runs, so it is as sensitive as the store; the reason to leave it in the work
 * directory is size, not safety. The work directory is sensitive too — see the README.
 */

process PREPROCESS {

    tag   "${meta.id}"
    label 'triage_asr'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*'
    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'derivatives/**',
               enabled: params.publish_derivatives

    input:
    tuple val(meta), path(audio)
    val   node_config

    output:
    tuple val(meta), path("store/segment.preprocess.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.preprocess.json"),    emit: verdict
    tuple val(meta), path("derivatives"),                      emit: derivatives
    path  "versions.yml",                                      emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def models = Triage.modelsArg(node_config.models)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store derivatives
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node PREPROCESS \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --config node-config.json \\
        ${models} \\
        --out store \\
        --derivatives derivatives \\
        ${replay}

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store derivatives
    printf '%s' ${cfg} > node-config.json

    triage-node --node PREPROCESS --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' \\
        --config node-config.json --out store --derivatives derivatives

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
