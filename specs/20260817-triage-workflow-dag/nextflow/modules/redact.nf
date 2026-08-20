/*
 * ============================================================================================
 * REDACT — the one node that may write to `release_dir`.
 *
 * This is the file where a mistake is a disclosure rather than a bug. Read the whole header.
 *
 * TWO PUBLISH ROOTS, TWO OUTPUT PREFIXES, AND THEY NEVER MIX
 * ---------------------------------------------------------
 *   store/*     -> params.store_dir     SENSITIVE. The new segment, the verdict. Element ids,
 *                                       assertion ids, categories, extents, provenance.
 *   release/*   -> params.release_dir   RELEASABLE. Redacted audio, redacted transcript, redacted
 *                                       figure, and a manifest carrying no ids.
 *
 * `redact.md`: "element ids are not shared between the store and a released artifact, because an id
 * that indexes both is a join key back to the PII". The chain the rule blocks is concrete: a
 * released artifact carrying `e-3f2a…` lets anyone holding the store look that element up, follow it
 * to the `word` element it refines, and read the text the redaction removed. The store deliberately
 * keeps the unredacted transcript, so the id is the whole attack.
 *
 * WHAT THE RELEASE MANIFEST DOES *NOT* CARRY, AND WHY
 * --------------------------------------------------
 * `redact.md`'s verdict carries `{ redactions_n, by_category, padding_ms, verified, survived[] }`.
 * That verdict is a product read by VERDICT — a store-domain reader — and it is published to
 * `store_dir`. The release manifest is a smaller thing: `release_id`, `redactions_n`, `padding_ms`,
 * `verified`. It omits
 *
 *   - element and assertion ids        the join key `redact.md` names;
 *   - extents                          the position of each redaction indexes the store's `pii`
 *                                      elements, which carry category and extent per finding;
 *   - by_category / survived           category plus position is most of a finding;
 *   - the recording's content hash and any store-side run id, for the same reason as ids: a
 *     shared key is a shared key regardless of what it is called. `redact.md` names element ids
 *     because they are the obvious case, not because they are the only one.
 *
 * The recording's *identity* does stay (the release path is keyed by `meta.id`). That is
 * unavoidable — a redacted derivative of a recording is inherently of that recording — and it is
 * not the same exposure: knowing which file this came from does not tell you where in it a name was
 * removed, or what the name was.
 *
 * THE GUARD IS PART OF THE NODE
 * -----------------------------
 * Before this task exits, `triage-node --node RELEASE_GUARD` sweeps every byte and every filename
 * under `release/` for the id pattern, for store-segment shapes, and for the manifest keys listed
 * above. A hit exits non-zero. That is a NODE ERROR, not a branch fail: it aborts, and because
 * `publishDir` runs only after a task succeeds, nothing reaches `release_dir`.
 *
 * A NODE FAIL IS NOT AN EMPTY RELEASE DIRECTORY
 * ---------------------------------------------
 * `redact.md`: a finding that survives verification is a `fail` and the artifact is not released.
 * So on `fail` this node writes NO `release/` files at all, `verdict.md` gives `release: withheld`,
 * and the exit status is still 0 because a branch fail is a value, not a status.
 *
 * VERIFICATION IS THE WEAKER CHECK AND THE VERDICT SAYS SO
 * -------------------------------------------------------
 * ASR on redacted audio may simply fail to transcribe a region that still contains intelligible
 * speech, so a clean re-scan is consistent with an incomplete redaction. The verdict carries
 * `verified: true` for "the re-scan found nothing", never "the audio is clean".
 * ============================================================================================
 */

process REDACT {

    tag   "${meta.id}"
    label 'triage_asr'

    // SENSITIVE root: the new segment and the verdict.
    publishDir path: { "${params.store_dir}/${meta.id}" },   mode: 'copy', pattern: 'store/*'

    // RELEASABLE root: REDACT's artifacts only. `saveAs` strips the `release/` prefix so the
    // published tree carries no hint of the two-root split, and `failOnError` makes a publish
    // problem loud rather than silent.
    publishDir path: { "${params.release_dir}/${meta.id}" }, mode: 'copy', pattern: 'release/*',
               saveAs: { fn -> fn.substring(fn.lastIndexOf('/') + 1) }, failOnError: true

    input:
    tuple val(meta), path(audio), path(derivatives, stageAs: 'derivatives'), path(store_in, stageAs: 'store_in/*')
    val   node_config

    output:
    tuple val(meta), path("store/segment.redact.*.jsonl"), emit: segment
    tuple val(meta), path("store/verdict.redact.json"),    emit: verdict
    tuple val(meta), path("release/*"),                    emit: release, optional: true
    path  "versions.yml",                                  emit: versions

    script:
    def cfg    = Triage.configArg(node_config)
    def models = Triage.modelsArg(node_config.models)
    def replay = params.replay ? "--replay ${file(params.replay)}" : ''
    """
    mkdir -p store store_in release
    printf '%s' ${cfg} > node-config.json

    triage-node \\
        --node REDACT \\
        --recording-id '${meta.id}' \\
        --audio '${audio}' \\
        --derivatives derivatives \\
        --store-in store_in \\
        --config node-config.json \\
        ${models} \\
        --out store \\
        --release release \\
        ${replay}

    # ---- the guard. Non-zero here aborts, and publishDir never runs. ----
    triage-node --node RELEASE_GUARD --release release --id-pattern '${params.redact.id_pattern}'

    # Belt and braces, independent of the Python: a raw byte sweep for the id shape.
    if grep -a -r -l -E '${params.redact.id_pattern}' release/ 2>/dev/null | grep -q . ; then
        echo "RELEASE GUARD: an element or assertion id reached the release directory." >&2
        exit 3
    fi

    # An empty release directory is a legitimate `fail`; remove it so the optional output does not
    # emit an empty tuple.
    rmdir release 2>/dev/null || true

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: \$(triage-node --version)
    END_VERSIONS
    """

    stub:
    def cfg = Triage.configArg(node_config)
    """
    mkdir -p store store_in release
    printf '%s' ${cfg} > node-config.json

    triage-node --node REDACT --stub --stub-scenario '${params.stub_scenario}' \\
        --recording-id '${meta.id}' --audio '${audio}' --derivatives derivatives \\
        --store-in store_in --config node-config.json --out store --release release

    # The guard runs in stub mode too. A stub that publishes an id-bearing file into the release
    # tree has taught the operator the wrong shape, and the shape is the point of the stub.
    triage-node --node RELEASE_GUARD --release release --id-pattern '${params.redact.id_pattern}'

    if grep -a -r -l -E '${params.redact.id_pattern}' release/ 2>/dev/null | grep -q . ; then
        echo "RELEASE GUARD: an element or assertion id reached the release directory." >&2
        exit 3
    fi

    rmdir release 2>/dev/null || true

    cat > versions.yml <<END_VERSIONS
    "${task.process}":
        triage-node: stub
    END_VERSIONS
    """
}
