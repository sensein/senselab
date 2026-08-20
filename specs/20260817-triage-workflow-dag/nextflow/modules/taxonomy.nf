/*
 * TAXONOMY — which kinds are in the recording.
 *
 * `taxonomy.md`: two kinds screened (airway, speech), the third — voice with no words — is the
 * residual and is *not screened*. Presence needs agreement across `min_families[kind]` eligible
 * families; absence needs unanimity; anything else is `undecided`.
 *
 * `min_families` HAS NO MEASURED VALUE (`benchmarks/open.md`). This node therefore does not pick
 * one. It evaluates the presence rule at EVERY admissible value — airway over {2, 3}, speech over
 * {1, 2} — and:
 *
 *   unanimous across the range  -> that state
 *   divergent across the range  -> `undecided`, reason `min_families_underived`
 *
 * That is not a workaround. `taxonomy.md` already defines `undecided` as "families disagree, or any
 * is unsure", and a rule whose own threshold is unlocated is unsure in exactly the same sense. The
 * alternative — a midpoint — would invent a decision the measurement does not contain, and would
 * report a confident state produced by an arbitrary constant.
 *
 * This node does not gate the branches. See the README: `verdict.md`'s "TAXONOMY said absent,
 * branch said pass -> flag" row is unreachable if a branch only runs when TAXONOMY admits its kind.
 */

process TAXONOMY {

    tag   "${meta.id}"
    label 'triage_model'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*'

    input:
    tuple val(meta), path(audio), path(derivatives, stageAs: 'derivatives'), path(store_in, stageAs: 'store_in/*')
    path  hints
    val   node_config

    output:
    tuple val(meta), path("store/segment.taxonomy.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.taxonomy.json"),    emit: verdict
    path  "versions.yml",                                    emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def models = Triage.modelsArg(node_config.models)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store store_in
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node TAXONOMY \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --derivatives derivatives \\
        --store-in store_in \\
        --hints '${hints}' \\
        --config node-config.json \\
        ${models} \\
        --out store \\
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

    triage-node --node TAXONOMY --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' --derivatives derivatives \\
        --store-in store_in --hints '${hints}' --config node-config.json --out store

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
