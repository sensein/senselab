/*
 * VERDICT — the last fold.
 *
 * `verdict.md`: two axes that never collapse into one.
 *
 *   triage   pass | flag | fail          does this recording need a human, and is it measurable
 *   release  releasable | withheld | not_assessed
 *
 * A BRANCH FAIL IS NOT A FILE FAIL. This is the rule everything else depends on, and it is why no
 * branch fail in this pipeline is ever an exit status. A cough recording has no speech, so SPEECH
 * failing is the expected outcome. Every node exits 0 when it ran; its conclusion is a string in a
 * verdict file, and this node reads those strings.
 *
 * A CONTRADICTION IS A FLAG. A branch outcome is read against what TAXONOMY said about its kind,
 * and two rows of that table exist only to be flagged:
 *
 *   TAXONOMY absent  + branch pass  -> the kind was present after all: resolve to present, and FLAG
 *   TAXONOMY present + branch fail  -> the screen found it, the branch found no subject: FLAG
 *
 * Those two rows are the reason this node exists, and they are also why the branches are NOT gated
 * on TAXONOMY anywhere in `main.nf`. A branch that only runs when TAXONOMY admits its kind can
 * never produce "absent + pass", so half the contradiction table would be dead code.
 *
 * This node needs the segments as well as the verdicts. `verdict.md` says it "reads verdicts rather
 * than elements", and then requires it to resolve the `kind` elements TAXONOMY wrote and to record
 * both TAXONOMY's assertion and its own resolution. That is an element write, so it gets the store.
 *
 * `remainder: true` on the upstream join means this node also runs for a recording ADMIT rejected,
 * where the only verdict in existence is ADMIT's own and there are no segments at all. Triage row 1
 * — "ADMIT failed -> fail, nothing ran" — is reachable only that way.
 */

process VERDICT {

    tag   "${meta.id}"
    label 'triage_cpu'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'store/*',
               saveAs: { fn -> fn.substring(fn.lastIndexOf('/') + 1) }

    input:
    tuple val(meta), path(verdicts, stageAs: 'verdicts/*'), path(store_in, stageAs: 'store_in/*')
    val   node_config

    output:
    tuple val(meta), path("store/verdict.file.json"),       emit: verdict
    tuple val(meta), path("store/segment.verdict.*.jsonl"), emit: segment
    path  "versions.yml",                                   emit: versions

    script:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store verdicts store_in
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node VERDICT \\
        --recording-id '${meta.id}' \\
        --verdicts-in verdicts \\
        --store-in store_in \\
        --config node-config.json \\
        --out store

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store verdicts store_in
    printf '%s' ${cfg} > node-config.json

    # The fold is pure logic over JSON, so the stub runs the REAL implementation. There is nothing
    # to fake, and faking it would make the one part of the graph most worth testing untested.
    triage-node \\
        --node VERDICT \\
        --recording-id '${meta.id}' \\
        --verdicts-in verdicts \\
        --store-in store_in \\
        --config node-config.json \\
        --out store

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub-runs-real-fold
    END_VERSIONS
    """
}
