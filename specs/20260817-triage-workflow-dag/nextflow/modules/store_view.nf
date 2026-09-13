/*
 * STORE_VIEW — the materialised view.
 *
 * `store.md` describes one store that any node may read in full. What actually exists on disk is a
 * set of single-author, content-addressed segments. This node assembles them into the thing the
 * document describes, and that assembly is where the reconciliation is either sound or not.
 *
 * WHY THE ASSEMBLY IS ORDER-INDEPENDENT
 * -------------------------------------
 *   1. Nothing is deleted and nothing is overwritten (`store.md`).
 *   2. An element id is "assigned once by the node that first proposed the element", so exactly one
 *      author writes any given element.
 *   3. An assertion never mutates its target; `contest` leaves both claims standing and `withdraw`
 *      is not deletion.
 *
 * Together those make the merged store a grow-only set: union is commutative, associative and
 * idempotent, so concatenating the segments in any order gives the same store. That is what makes
 * "the store as of node N" a pure function of the SET of upstream segments rather than of the
 * sequence in which they were written — and therefore something Nextflow can hash and cache.
 *
 * WHAT THIS NODE EMITS
 * --------------------
 *   view/store.jsonl     every record from every segment, concatenated, with a header naming the
 *                        segments folded and their content hashes
 *   view/elements.json   one entry per element, with its assertions in author order and the
 *                        contested pairs left contested. This is A fold, not THE fold: `store.md`
 *                        says the current view is a reader's choice, so this file names the fold it
 *                        applied (`fold: "all_claims"`) and does not pretend to be canonical
 *   view/integrity.json  duplicate element ids across authors, assertions whose target is absent,
 *                        elements or assertions with no model revision where a model was involved
 *
 * `integrity.json` findings are reported, not enforced — except one. An element id claimed by two
 * different authors breaks property 2 above and therefore breaks the commutativity the whole
 * reconciliation rests on, so it exits non-zero.
 *
 * SENSITIVE. The view is the store, and after SPEECH the store holds an unredacted transcript.
 */

process STORE_VIEW {

    tag   "${meta.id}"
    label 'triage_cpu'

    publishDir path: { "${params.store_dir}/${meta.id}" }, mode: 'copy', pattern: 'view/*'

    input:
    tuple val(meta), path(store_in, stageAs: 'store_in/*')
    val   node_config

    output:
    tuple val(meta), path("view/store.jsonl"),     emit: view
    tuple val(meta), path("view/elements.json"),   emit: elements
    tuple val(meta), path("view/integrity.json"),  emit: integrity
    path  "versions.yml",                          emit: versions

    script:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p view store_in
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node STORE_VIEW \\
        --recording-id '${meta.id}' \\
        --store-in store_in \\
        --config node-config.json \\
        --out view

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p view store_in
    printf '%s' ${cfg} > node-config.json

    triage-node --node STORE_VIEW --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --store-in store_in --config node-config.json --out view

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
