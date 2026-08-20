/*
 * ADMIT — is this file measurable at all.
 *
 * `admit.md`: two outcomes, `pass` or `fail`. No `flag`, no models, no thresholds, no second
 * version of the audio. The only rejections are degenerate: decode failure, all-zero, constant.
 *
 * ADMIT writes NO segment. Its product is a decoded signal, and `store.md`'s element kinds are
 * span / word / speaker / interval / measurement — none of which a decoded waveform is. So the
 * store begins at PREPROCESS and ADMIT contributes a verdict only. Appending a "level" element
 * here just to have something to append would be exactly the accumulation of unread ports that
 * `admit.md` refuses.
 *
 * The `audio` output is `optional: true`. That is how a `fail` stops the graph without an exit
 * status: no file, no emission, nothing downstream runs — and the verdict still reaches VERDICT so
 * that `verdict.md` triage row 1 can fire.
 */

process ADMIT {

    tag   "${meta.id}"
    label 'triage_cpu'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*'

    input:
    tuple val(meta), path(recording)
    val   node_config

    output:
    tuple val(meta), path("store/verdict.admit.json"),    emit: verdict
    tuple val(meta), path("admitted/audio.decoded.wav"),  emit: audio, optional: true
    path  "versions.yml",                                 emit: versions

    script:
    def cfg = Triage.configArg(node_config)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store admitted
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node ADMIT \\
        --recording-id '${meta.id}' \\
        --audio '${recording}' \\
        --config node-config.json \\
        --out store \\
        --decoded admitted/audio.decoded.wav \\
        ${replay}

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store admitted
    printf '%s' ${cfg} > node-config.json

    triage-node --node ADMIT --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${recording}' \\
        --config node-config.json --out store --decoded admitted/audio.decoded.wav

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
