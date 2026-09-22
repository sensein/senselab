"""One speaker embedding per subject: what is admitted, what is refused, and what a row carries."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pytest
import soundfile

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage import speaker_vectors as sv

PROFILE = Path(sv.__file__).parent / "data" / "speaker_vector_profile" / "2026-09-22.yaml"


# ----------------------------------------------------------------------------- a synthetic store


def _entity(entity_id: str, prov_type: str, extent: Any, **attributes: Any) -> str:  # noqa: ANN401 -- store values
    return json.dumps(
        {"record": "entity", "id": entity_id, "prov_type": prov_type, "extent": extent, "attributes": attributes}
    )


def _relation(relation: str, source: str, target: str) -> str:
    return json.dumps({"record": "relation", "relation": relation, "source": source, "target": target})


def build_recording(
    root: Path,
    stem: str = "sub-a1_ses-b2_task-prolonged-vowel",
    timestamp: str = "20260920-030735",
    duration_s: float = 12.0,
    sampling_rate: int = 16000,
    *,
    extents: tuple[tuple[float, float, str], ...] = ((1.0, 9.0, "voice"),),
    duplicate_extent: bool = False,
    invalidate_extent: bool = False,
    with_stream_file: bool = True,
) -> Path:
    """A run directory carrying a plain stream and whatever task extents a test needs.

    Args:
        root: Where the ``sub-*/ses-*/<stem>_<timestamp>`` tree goes.
        stem: The BIDS stem.
        timestamp: The launch timestamp in the run directory's name.
        duration_s: Length of the plain stream.
        sampling_rate: Its sampling rate.
        extents: ``(start_s, end_s, family)`` per task extent to mint.
        duplicate_extent: Whether to write each extent twice under different entity ids.
        invalidate_extent: Whether to invalidate the first extent.
        with_stream_file: Whether the plain flac exists on disk.

    Returns:
        The run root, the directory holding ``run/``.
    """
    participant, session = stem.split("_")[0], stem.split("_")[1]
    run_root = root / participant / session / f"{stem}_{timestamp}"
    run_dir = run_root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)

    if with_stream_file:
        rng = np.random.default_rng(abs(hash(stem)) % (2**32))
        samples = rng.normal(0.0, 0.05, int(duration_s * sampling_rate)).astype("float32")
        soundfile.write(run_dir / "streams" / "plain.flac", samples, sampling_rate)

    lines = [
        _entity(
            "stream-plain",
            "stream",
            [0.0, duration_s],
            name="plain",
            path="streams/plain.flac",
            sampling_rate=sampling_rate,
            channels=1,
        )
    ]
    for index, (start_s, end_s, family) in enumerate(extents):
        lines.append(_entity(f"span-{index}", "span", [start_s, end_s], role="task_extent", family=family))
        if duplicate_extent:
            lines.append(_entity(f"span-{index}-dup", "span", [start_s, end_s], role="task_extent", family=family))
        # A sibling span that is not a task extent, to prove the role filter bites.
        lines.append(_entity(f"span-{index}-other", "span", [start_s, end_s], role="phonation", family=family))
    if invalidate_extent and extents:
        lines.append(_relation("wasInvalidatedBy", "span-0", "activity-x"))
    (run_dir / "store.jsonl").write_text("\n".join(lines) + "\n")
    return run_root


def build_subject(root: Path, subject: str, spans: tuple[tuple[str, float, float], ...]) -> None:
    """Write one recording per span for a subject.

    Args:
        root: The corpus root.
        subject: The ``sub-*`` label.
        spans: ``(task, start_s, end_s)`` per recording.
    """
    for index, (task, start_s, end_s) in enumerate(spans):
        build_recording(
            root,
            stem=f"{subject}_ses-s{index}_task-{task}",
            duration_s=end_s + 2.0,
            extents=((start_s, end_s, "speech"),),
        )


# --------------------------------------------------------------------------------- the extents


def test_a_task_extent_is_read_with_its_seconds_family_and_audio_path(tmp_path: Path) -> None:
    """The span the branch minted comes back as one Extent pointing at the plain stream."""
    run_root = build_recording(tmp_path, extents=((1.5, 7.25, "voice"),))
    (extent,) = sv.read_task_extents(run_root)
    assert (extent.start_s, extent.end_s) == (1.5, 7.25)
    assert extent.duration_s == pytest.approx(5.75)
    assert extent.family == "voice"
    assert extent.subject == "sub-a1"
    assert extent.session == "ses-b2"
    assert extent.task == "task-prolonged-vowel"
    assert extent.audio_path == run_root / "run" / "streams" / "plain.flac"


def test_a_span_that_is_not_a_task_extent_is_not_read(tmp_path: Path) -> None:
    """Only role == task_extent counts; a phonation span sharing the interval is ignored."""
    run_root = build_recording(tmp_path, extents=((1.0, 5.0, "voice"),))
    assert len(sv.read_task_extents(run_root)) == 1


def test_the_same_interval_written_twice_yields_one_extent(tmp_path: Path) -> None:
    """The replayed stores write the span twice; a speaker must not be charged for it twice."""
    run_root = build_recording(tmp_path, extents=((1.0, 5.0, "speech"),), duplicate_extent=True)
    assert len(sv.read_task_extents(run_root)) == 1


def test_an_invalidated_extent_is_dropped(tmp_path: Path) -> None:
    """A wasInvalidatedBy relation removes the span from the live view."""
    run_root = build_recording(tmp_path, extents=((1.0, 5.0, "speech"),), invalidate_extent=True)
    assert sv.read_task_extents(run_root) == []


def test_a_recording_that_minted_no_extent_contributes_nothing(tmp_path: Path) -> None:
    """Absence of the span is the branch's record, not an error and not a whole-file fallback."""
    run_root = build_recording(tmp_path, extents=())
    assert sv.read_task_extents(run_root) == []


def test_a_run_directory_with_a_collision_suffix_still_yields_its_identity(tmp_path: Path) -> None:
    """prepare_run_layout appends -1 on a same-second collision; the stem must survive it."""
    run_root = build_recording(tmp_path, timestamp="20260920-030735-1", extents=((1.0, 5.0, "speech"),))
    (extent,) = sv.read_task_extents(run_root)
    assert extent.subject == "sub-a1"
    assert extent.task == "task-prolonged-vowel"


# ------------------------------------------------------------------------------------ the floor


def test_the_floor_comes_from_the_profile_and_not_from_a_literal() -> None:
    """A refusal threshold lives in data/ with its derivation, never as a number in the code."""
    assert PROFILE.exists()
    floor = sv.load_min_extent_s()
    assert floor > 0
    assert f"{floor}" in PROFILE.read_text()


def test_a_missing_profile_raises_rather_than_substituting_a_number(tmp_path: Path) -> None:
    """An unfitted floor reported as measured is worse than a crash."""
    with pytest.raises(ValueError, match="missing from this install"):
        sv.load_min_extent_s(tmp_path / "absent.yaml")


def test_a_profile_with_the_wrong_schema_version_raises(tmp_path: Path) -> None:
    """A profile the code cannot interpret must not be read optimistically."""
    path = tmp_path / "p.yaml"
    path.write_text("schema_version: '99'\nmin_extent_s: 1.0\n")
    with pytest.raises(ValueError, match="schema_version"):
        sv.load_min_extent_s(path)


def test_an_extent_below_the_floor_is_refused_and_counted(tmp_path: Path) -> None:
    """A refused extent is excluded from the estimate and visible in the report."""
    floor = sv.load_min_extent_s()
    build_subject(tmp_path, "sub-short", (("a", 0.5, 0.5 + floor / 2),))
    grouped, report = sv.gather(tmp_path)
    assert grouped == {}
    assert report.extents_refused_short == 1
    assert report.subjects_all_refused == ["sub-short"]
    assert report.subjects_without_extent == []


def test_a_subject_that_minted_no_extent_is_reported_apart_from_one_that_was_refused(tmp_path: Path) -> None:
    """Nothing to embed and everything too short are different facts about a speaker."""
    build_recording(tmp_path, stem="sub-none_ses-s0_task-x", extents=())
    grouped, report = sv.gather(tmp_path)
    assert grouped == {}
    assert report.subjects_without_extent == ["sub-none"]
    assert report.subjects_all_refused == []


def test_an_extent_whose_stream_is_missing_is_refused_and_counted(tmp_path: Path) -> None:
    """A span with no audio behind it must not silently become a thinner estimate."""
    build_recording(
        tmp_path,
        stem="sub-nofile_ses-s0_task-x",
        duration_s=20.0,
        extents=((1.0, 15.0, "speech"),),
        with_stream_file=False,
    )
    grouped, report = sv.gather(tmp_path)
    assert grouped == {}
    assert report.extents_missing_audio == 1


# ------------------------------------------------------------------------------------- the scan


def test_a_subject_lands_in_one_shard_and_the_shards_cover_the_tree(tmp_path: Path) -> None:
    """Four shards partition the subjects exactly once each."""
    for index in range(12):
        build_subject(tmp_path, f"sub-{index:02d}", (("x", 1.0, 15.0),))
    seen: list[str] = []
    for shard in range(4):
        grouped, _ = sv.gather(tmp_path, shard, 4)
        seen.extend(grouped)
    assert sorted(seen) == sorted(f"sub-{index:02d}" for index in range(12))


def test_shard_assignment_depends_only_on_the_subject(tmp_path: Path) -> None:
    """A growing tree must never reshuffle a subject into another worker."""
    assert sv.shard_of("sub-abc", 64) == sv.shard_of("sub-abc", 64)
    assert sv.shard_of("sub-abc", 1) == 0


def test_a_rerun_of_one_recording_contributes_once_at_its_latest_timestamp(tmp_path: Path) -> None:
    """Two runs of the same stem are one recording, not two extents from one reading."""
    build_recording(tmp_path, stem="sub-r_ses-s_task-x", timestamp="20260920-010000", extents=((1.0, 9.0, "speech"),))
    build_recording(tmp_path, stem="sub-r_ses-s_task-x", timestamp="20260920-020000", extents=((2.0, 9.0, "speech"),))
    grouped, _ = sv.gather(tmp_path)
    (extent,) = grouped["sub-r"]
    assert extent.start_s == 2.0


def test_every_admitted_extent_is_counted_in_the_report(tmp_path: Path) -> None:
    """The report's arithmetic must close: what was seen, what carried a span, what was admitted."""
    build_subject(tmp_path, "sub-many", (("a", 1.0, 15.0), ("b", 1.0, 20.0)))
    build_recording(tmp_path, stem="sub-many_ses-s9_task-c", extents=())
    grouped, report = sv.gather(tmp_path)
    assert len(grouped["sub-many"]) == 2
    assert report.recordings_seen == 3
    assert report.recordings_with_extent == 2
    assert report.extents_admitted == 2
    assert report.subjects_considered == 1


# ---------------------------------------------------------------------------------- the schema


def test_the_identity_the_vector_and_its_support_come_first() -> None:
    """A reader opening the parquet sees who, what, and how much before any diagnostic."""
    names = sv.schema().names
    assert names[:6] == ["speaker_id", "vector", "dim", "n_extents", "n_recordings", "extent_seconds"]


def test_the_schema_version_is_in_the_metadata_and_on_every_row() -> None:
    """A file and a row both have to say which schema produced them."""
    metadata = sv.schema().metadata
    assert metadata[b"senselab.speaker_vectors.schema_version"] == str(sv.SCHEMA_VERSION).encode()
    assert "schema_version" in sv.schema().names


def test_the_vector_is_a_list_of_doubles_and_not_a_blob() -> None:
    """The vector is small and read whole; there is nothing to quantise away."""
    assert sv.schema().field("vector").type == pa.list_(pa.float64())


def test_the_per_extent_columns_are_parallel_lists_of_one_length() -> None:
    """Every per-extent column is indexed by the same position, so a reader can zip them."""
    row = _row(n_extents=3)
    table = sv.to_table([row])
    lengths = {
        name: len(table.column(name)[0].as_py())
        for name in table.schema.names
        if name.startswith("extent_") and name != "extent_seconds"
    }
    assert set(lengths.values()) == {3}


def test_a_key_a_row_does_not_carry_becomes_null_and_not_a_default() -> None:
    """An absent statistic is absent; a substituted zero would read as a measurement."""
    row = _row()
    row.pop("auc_same_extent_vs_diff_extent")
    table = sv.to_table([row])
    assert table.column("auc_same_extent_vs_diff_extent")[0].as_py() is None


def test_two_shards_share_one_schema_so_a_merge_is_a_concat() -> None:
    """Shards must not need casting to be joined."""
    assert sv.to_table([_row()]).schema == sv.to_table([_row(n_extents=2)]).schema
    assert sv.schema().empty_table().schema == sv.to_table([_row()]).schema


# ---------------------------------------------------------------------------------------- a row


class _Counts:
    def __init__(self, per_file: dict[str, int]) -> None:
        self.vectors_per_file = per_file
        self.n_scored = sum(per_file.values())
        self.n_effective = float(self.n_scored) / 2.0


class _Stats:
    min = q05 = q25 = q50 = q75 = q95 = max = mean = sd = 0.9


class _Block:
    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401 -- a stand-in for a pydantic block
        self.__dict__.update(kwargs)


def _embedding(extent_ids: list[str]) -> Any:  # noqa: ANN401 -- a stand-in TargetSpeakerEmbedding
    per_file = {eid: 4 for eid in extent_ids}
    distribution = _Block(
        counts=_Counts(per_file),
        nulls=_Block(rbar_null=0.5, participation_ratio_null=10.0, cos_sd_null=0.07, auc_null=0.5),
        cos_to_centroid_loo=_Stats(),
        rbar=0.8,
        spectrum=_Block(participation_ratio=3.0, pc1_share_centred=0.4, eigenvalue_shares_top5=[]),
        within_file={},
        cross_file=_Block(
            cos_file_centroid_to_pooled={eid: 0.9 for eid in extent_ids},
            file_centroid_pairwise_cos=_Stats() if len(extent_ids) > 1 else None,
        ),
        file_effect=_Block(auc_same_file_vs_diff_file=0.7 if len(extent_ids) > 1 else None),
        centroid_robustness=_Block(
            cos_mean_vs_trimmed10=0.99,
            cos_mean_vs_medoid=0.95,
            leave_one_file_out_cos={eid: 0.98 for eid in extent_ids},
        ),
    )
    provenance = _Block(
        model_id=sv.MODEL_ID,
        model_commit_sha="0" * 40,
        unresolved_reason=None,
        method=sv.AGGREGATOR,
        source_files=list(extent_ids),
        window_s=sv.WINDOW_S,
        hop_s=sv.HOP_S,
        n_windows_used=4 * len(extent_ids),
        n_windows_dropped=0,
        extraction_failures={},
    )
    return _Block(vector=np.ones(sv.EMBEDDING_DIM), provenance=provenance, distribution=distribution)


def _extents(n: int) -> list[sv.Extent]:
    return [
        sv.Extent(
            subject="sub-a1",
            session=f"ses-s{i}",
            task=f"task-t{i}",
            stem=f"sub-a1_ses-s{i}_task-t{i}",
            run_dir=f"sub-a1_ses-s{i}_task-t{i}_20260920-03073{i}",
            audio_path=Path(f"/tmp/{i}.flac"),
            start_s=1.0,
            end_s=9.0,
            family="speech",
            production=None,
        )
        for i in range(n)
    ]


def _row(n_extents: int = 1, root: Optional[Path] = None) -> dict[str, Any]:
    extents = _extents(n_extents)
    return sv.row_for("sub-a1", extents, _embedding([e.extent_id for e in extents]), root or Path("/corpus"))


def test_a_row_counts_the_extents_recordings_and_seconds_that_went_in(tmp_path: Path) -> None:
    """The three numbers a reader needs to weight a speaker's vector."""
    row = _row(n_extents=3)
    assert row["n_extents"] == 3
    assert row["n_recordings"] == 3
    assert row["n_sessions"] == 3
    assert row["extent_seconds"] == pytest.approx(24.0)
    assert row["dim"] == sv.EMBEDDING_DIM


def test_a_row_names_the_recordings_that_contributed(tmp_path: Path) -> None:
    """Provenance a reader needs to go back to the audio."""
    row = _row(n_extents=2)
    assert row["extent_stem"] == ["sub-a1_ses-s0_task-t0", "sub-a1_ses-s1_task-t1"]
    assert row["extent_start_s"] == [1.0, 1.0]
    assert row["extent_end_s"] == [9.0, 9.0]


def test_a_row_carries_the_model_at_a_resolved_commit_and_never_a_ref() -> None:
    """A ref in the commit column would be provenance that is confidently wrong."""
    row = _row()
    assert row["model_id"] == sv.MODEL_ID
    assert len(row["model_commit_sha"]) == 40
    assert row["unresolved_reason"] is None


def test_a_single_extent_speaker_has_no_cross_extent_statistic(tmp_path: Path) -> None:
    """With one extent there is no pair, so the agreement columns are null rather than invented."""
    row = _row(n_extents=1)
    assert row["cos_extent_pairwise_q50"] is None
    assert row["auc_same_extent_vs_diff_extent"] is None
    assert row["n_extents"] == 1


def test_the_pooling_and_window_grid_are_recorded_on_the_row() -> None:
    """Two rows pooled differently are not comparable, so the rule travels with the vector."""
    row = _row()
    assert row["method"] == sv.AGGREGATOR
    assert (row["window_s"], row["hop_s"]) == (sv.WINDOW_S, sv.HOP_S)


def test_a_subject_whose_estimate_raises_is_recorded_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opaque failure count would hide the one case the floor does not cover."""
    build_subject(tmp_path, "sub-boom", (("a", 1.0, 15.0),))

    def _boom(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise ValueError("need at least 2 non-zero vectors to describe a distribution; got 1")

    monkeypatch.setattr(sv, "embed_subject", _boom)
    rows, report = sv.scan(tmp_path)
    assert rows == []
    assert report.subjects_written == 0
    assert "sub-boom" in report.subjects_failed
    assert "2 non-zero vectors" in report.subjects_failed["sub-boom"]


def test_one_writer_of_a_duplicated_interval_surviving_keeps_the_extent(tmp_path: Path) -> None:
    """The store writes the span twice; invalidating one copy must not drop the interval."""
    run_root = build_recording(tmp_path, extents=((1.0, 9.0, "speech"),), duplicate_extent=True)
    store = run_root / "run" / "store.jsonl"
    store.write_text(store.read_text() + _relation("wasInvalidatedBy", "span-0", "activity-x") + "\n")
    (extent,) = sv.read_task_extents(run_root)
    assert (extent.start_s, extent.end_s) == (1.0, 9.0)


def test_a_long_extent_does_not_outvote_a_short_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pooling is extent-equal: eight windows of one extent weigh what two of another do."""
    import senselab.audio.tasks.speaker_embeddings.windowing as windowing
    from senselab.audio.tasks.speaker_embeddings.windowing import WindowEmbedding

    build_recording(tmp_path, stem="sub-w_ses-s0_task-long", duration_s=20.0, extents=((1.0, 19.0, "speech"),))
    build_recording(tmp_path, stem="sub-w_ses-s1_task-short", duration_s=8.0, extents=((1.0, 6.0, "speech"),))
    grouped, _ = sv.gather(tmp_path)
    extents = sorted(grouped["sub-w"], key=lambda e: e.duration_s)

    # Two orthogonal directions, one per extent, with eight windows against two.
    east, north = np.zeros(sv.EMBEDDING_DIM), np.zeros(sv.EMBEDDING_DIM)
    east[0], north[1] = 1.0, 1.0
    plan = {extents[0].extent_id: (north, 2), extents[1].extent_id: (east, 8)}
    by_duration = {round(e.duration_s, 3): plan[e.extent_id] for e in extents}

    def _windows(*, audio: Audio, **_kwargs: object) -> dict[str, list[WindowEmbedding]]:
        seconds = round(audio.waveform.shape[-1] / audio.sampling_rate, 3)
        direction, count = by_duration[seconds]
        return {sv.MODEL_ID: [WindowEmbedding(float(i), float(i) + 2.0, direction) for i in range(count)]}

    monkeypatch.setattr(windowing, "extract_per_window_embeddings", _windows)
    row = sv.embed_subject("sub-w", extents, tmp_path)

    vector = np.asarray(row["vector"])
    # Equal weighting puts the centroid exactly between the two directions; window weighting
    # would pull it to 0.97/0.24 in favour of the eight-window extent.
    assert vector[0] == pytest.approx(0.7071, abs=1e-3)
    assert vector[1] == pytest.approx(0.7071, abs=1e-3)
    assert row["method"] == "recording_equal_spherical_mean"
    assert row["n_windows_used"] == 10


def test_two_overlapping_extents_of_one_recording_cast_one_vote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replayed AIRWAY branch mints two nested spans; that is one recording, not two."""
    import senselab.audio.tasks.speaker_embeddings.windowing as windowing
    from senselab.audio.tasks.speaker_embeddings.windowing import WindowEmbedding

    # One recording minting two near-identical extents, and one minting a single extent.
    build_recording(
        tmp_path,
        stem="sub-v_ses-s0_task-respiration",
        duration_s=30.0,
        extents=((2.24, 25.74, "airway"), (2.24, 26.69, "airway")),
    )
    build_recording(tmp_path, stem="sub-v_ses-s1_task-harvard", duration_s=12.0, extents=((1.0, 9.0, "speech"),))
    grouped, _ = sv.gather(tmp_path)
    extents = sorted(grouped["sub-v"], key=lambda e: e.stem)
    assert len(extents) == 3

    east, north = np.zeros(sv.EMBEDDING_DIM), np.zeros(sv.EMBEDDING_DIM)
    east[0], north[1] = 1.0, 1.0

    def _windows(*, audio: Audio, **_kwargs: object) -> dict[str, list[WindowEmbedding]]:
        # The two airway cuts are both over 20 s; the speech cut is 8 s.
        seconds = audio.waveform.shape[-1] / audio.sampling_rate
        direction = east if seconds > 20.0 else north
        return {sv.MODEL_ID: [WindowEmbedding(0.0, 2.0, direction), WindowEmbedding(1.0, 3.0, direction)]}

    monkeypatch.setattr(windowing, "extract_per_window_embeddings", _windows)
    vector = np.asarray(sv.embed_subject("sub-v", extents, tmp_path)["vector"])

    # Two recordings, one vote each: the centroid sits exactly between the two directions.
    # Per-extent weighting would have given the airway recording two of three votes (0.89/0.45).
    assert vector[0] == pytest.approx(0.7071, abs=1e-3)
    assert vector[1] == pytest.approx(0.7071, abs=1e-3)
