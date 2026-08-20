/*
 * VOICE branch — vocalic activity that is neither airway nor speech.
 *
 * `branch-voice.md`: it measures, it does not classify. No classifier in the screening set can name
 * a member of this kind, so the branch owns no label space and loads no model.
 *
 * Visibility: PREPROCESS + TAXONOMY + AIRWAY + SPEECH. Step 1's residual is "intervals with energy,
 * minus airway-labelled spans, minus speech spans", which is a fold over what the OTHER TWO
 * BRANCHES asserted. `branch-voice.md` says so explicitly and treats it as the store's chief
 * benefit: the earlier design named a `residual_windows` input with no producer anywhere in the
 * graph, and the store gives it one. The cost is that this branch cannot start until both others
 * have finished.
 *
 * THE GATE HAS NO VALUE. `branch-voice.md` step 2 and `benchmarks/open.md`: periodicity anywhere in
 * (0.44, 0.933), RMS anywhere in (0.0007, 0.0161) — a factor of 2.1 and a factor of 23, on one
 * recording, with the derivation slot deliberately empty.
 *
 * So the gate is evaluated at both endpoints of each interval and the outcome is interval-valued:
 *
 *   passes at both endpoints -> voiced run
 *   fails at both endpoints  -> not a run
 *   differs                  -> the run is recorded with `gate_undetermined`, and the branch flags
 *
 * `branch-voice.md` already lists exactly that flag — "the gate's parameters are still un-derived
 * and a run sits near the interval's edge" — so this is the spec's own rule made executable rather
 * than a new one. A midpoint would have produced a confident boundary out of a factor-of-23 range.
 */

process VOICE {

    tag   "${meta.id}"
    label 'triage_dsp'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*'

    input:
    tuple val(meta), path(audio), path(derivatives, stageAs: 'derivatives'), path(store_in, stageAs: 'store_in/*')
    path  hints
    val   node_config

    output:
    tuple val(meta), path("store/segment.voice.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.voice.json"),    emit: verdict
    path  "versions.yml",                                 emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store store_in
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node VOICE \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --derivatives derivatives \\
        --store-in store_in \\
        --hints '${hints}' \\
        --config node-config.json \\
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

    triage-node --node VOICE --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' --derivatives derivatives \\
        --store-in store_in --hints '${hints}' --config node-config.json --out store

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
