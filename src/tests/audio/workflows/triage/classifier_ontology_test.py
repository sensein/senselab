"""The classifier-ontology profile: every HeAR label resolves, and every id it names exists.

The hand-written map this replaced covered two of HeAR's eight labels, so the other six could never
be corroborated. These tests pin the resolution of all eight, the subtree semantics that make
``Throat clearing`` corroborate ``Cough``, and the profile's internal closure. See
``specs/20260910-classifier-ontology-mapping/design.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.tasks.health_acoustics.hear import HEAR_EVENT_LABELS
from senselab.audio.workflows.triage.classifier_ontology import (
    PROFILE_DIR,
    PROFILE_VERSION,
    airway_audioset_labels,
    airway_hear_labels,
    audioset_labels_for_group,
    audioset_labels_under_roots,
    canonical_names,
    corroboration_sets,
    hear_labels,
    hear_labels_in_group,
    hear_labels_under_roots,
    load_classifier_ontology,
    unemittable_by_yamnet,
)
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.labels import (
    AUDIOSET_AIRWAY,
    AUDIOSET_BREATH,
    AUDIOSET_COUGH,
    FAMILIES,
    HEAR_AIRWAY,
    HEAR_BREATH,
    HEAR_COUGH,
    LABEL_SETS,
)


def load_airway_roots() -> list[str]:
    """The airway ontology roots the packaged configuration names.

    Returns:
        The root display names.
    """
    return [str(root) for root in load_triage_config().require("taxonomy.airway_ontology_roots")]


class TestTheProfileIsShipped:
    """A profile that is absent, stale or unparseable takes airway corroboration with it."""

    def test_a_profile_is_bundled(self) -> None:
        """The generated file is packaged, not left to be built on the operator's machine."""
        assert sorted(PROFILE_DIR.glob("*.json")), f"no profile in {PROFILE_DIR}"

    def test_the_bundled_profile_loads(self) -> None:
        """Loading validates; a profile that fails validation raises rather than returning empty."""
        assert load_classifier_ontology()["profile_version"] == PROFILE_VERSION

    def test_every_source_is_pinned_by_commit_and_digest(self) -> None:
        """An unpinned provenance is worse than none: a moved branch would change the map silently."""
        for key, source in load_classifier_ontology()["sources"].items():
            assert len(source["commit"]) == 40, key
            assert len(source["sha256"]) == 64, key
            assert source["licence"] and source["attribution"], key
            assert source["commit"] in source["url"], key

    def test_the_profile_names_hears_own_eight_labels(self) -> None:
        """The authority for HeAR's label set is the detector's graph order, not a docstring."""
        assert hear_labels() == HEAR_EVENT_LABELS


class TestEveryHearLabelResolves:
    """The defect: six of eight labels had no entry and so could never be corroborated."""

    @pytest.mark.parametrize("label", HEAR_EVENT_LABELS)
    def test_the_label_has_a_non_empty_corroboration_set(self, label: str) -> None:
        """A label whose set is empty is a label the branch can never confirm."""
        assert corroboration_sets()[label]

    def test_cough_is_corroborated_by_its_own_node_and_its_only_child(self) -> None:
        """AudioSet puts `Throat clearing` under `Cough`, so a throat clear corroborates a cough."""
        assert corroboration_sets()["Cough"] == frozenset({"Cough", "Throat clearing"})

    def test_throat_clear_is_corroborated_by_audioset_throat_clearing(self) -> None:
        """HeAR's `Throat Clear` and AudioSet's `Throat clearing` differ only in spelling."""
        assert "Throat clearing" in corroboration_sets()["Throat Clear"]

    def test_baby_cough_is_corroborated_by_cough(self) -> None:
        """AudioSet has no infant cough class, so `Baby Cough` maps onto the adult node."""
        assert "Cough" in corroboration_sets()["Baby Cough"]
        assert load_classifier_ontology()["mapping"]["Baby Cough"]["note"]

    def test_snore_is_corroborated_by_snoring(self) -> None:
        """HeAR's `Snore` and AudioSet's `Snoring` differ only in spelling."""
        assert corroboration_sets()["Snore"] == frozenset({"Snoring"})

    def test_breathe_covers_the_breathing_subtree(self) -> None:
        """Wheeze, Snoring, Gasp, Pant and Snort are all children of Breathing."""
        assert corroboration_sets()["Breathe"] == frozenset({"Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort"})

    def test_a_label_outside_every_set_corroborates_nothing(self) -> None:
        """`Rain` is a real AudioSet class and is in no HeAR label's set."""
        assert not any("Rain" in names for names in corroboration_sets().values())

    def test_sigh_corroborates_no_label(self) -> None:
        """The hand map had Sigh corroborate Breathe; AudioSet puts Sigh under Human voice."""
        assert not any("Sigh" in names for names in corroboration_sets().values())


class TestTheOverlapIsRecordedNotDeduplicated:
    """Snoring is a descendant of Breathing, so two HeAR labels claim it. That is a fact, not a bug."""

    def test_snore_and_breathe_share_snoring(self) -> None:
        """Both sets contain it; neither is trimmed to make them disjoint."""
        sets = corroboration_sets()
        assert "Snoring" in sets["Snore"] and "Snoring" in sets["Breathe"]

    def test_the_profile_records_that_overlap(self) -> None:
        """A silent overlap would read as two independent corroborations of one AudioSet class."""
        overlaps = {tuple(entry["labels"]): entry["shared_names"] for entry in load_classifier_ontology()["overlaps"]}
        assert overlaps[("Snore", "Breathe")] == ["Snoring"]

    def test_the_cough_group_overlaps_are_recorded_too(self) -> None:
        """`Cough`, `Baby Cough` and `Throat Clear` all reach `Throat clearing`."""
        overlaps = {tuple(entry["labels"]) for entry in load_classifier_ontology()["overlaps"]}
        assert {("Cough", "Baby Cough"), ("Cough", "Throat Clear"), ("Baby Cough", "Throat Clear")} <= overlaps


class TestWhatYamnetCannotEmit:
    """YAMNet carries 521 of AudioSet's 527 classes. A class it lacks can never fire."""

    def test_the_two_class_lists_differ_by_six(self) -> None:
        """AST's 527 minus YAMNet's 521; the difference is recorded, not assumed to be empty."""
        profile = load_classifier_ontology()["audioset"]
        assert profile["released_class_count"] == 527
        assert profile["yamnet_class_count"] == 521
        assert profile["released_not_in_yamnet"] == [
            "Battle cry",
            "Female singing",
            "Female speech, woman speaking",
            "Funny music",
            "Male singing",
            "Male speech, man speaking",
        ]

    def test_only_speechs_set_carries_a_class_yamnet_cannot_report(self) -> None:
        """The gendered speech classes are in Speech's subtree and in AST's list but not YAMNet's."""
        assert unemittable_by_yamnet() == {"Speech": ("Male speech, man speaking", "Female speech, woman speaking")}

    def test_no_airway_label_depends_on_an_unemittable_class(self) -> None:
        """Every class corroborating a cough or a breath is one YAMNet can actually report."""
        assert not set(unemittable_by_yamnet()) & set(HEAR_COUGH + HEAR_BREATH)


class TestTheProfileIsInternallyConsistent:
    """The regression guard: a hand-edited or half-regenerated profile must not load."""

    def test_every_referenced_audioset_id_exists_in_the_ontology(self) -> None:
        """A mapping onto an id the node table does not hold is a mapping onto nothing."""
        profile = load_classifier_ontology()
        classes = profile["audioset"]["classes"]
        for label, entry in profile["mapping"].items():
            for node_id in entry["audioset_ids"] + entry["corroborating_ids"]:
                assert node_id in classes, f"{label} -> {node_id}"

    def test_every_corroborating_name_matches_its_id(self) -> None:
        """The names are what the classifiers report; a name out of step with its id is a silent miss."""
        profile = load_classifier_ontology()
        classes = profile["audioset"]["classes"]
        for entry in profile["mapping"].values():
            names = [classes[node_id]["name"] for node_id in entry["corroborating_ids"]]
            assert names == entry["corroborating_names"]

    def test_every_corroborating_set_is_the_subtree_of_its_roots(self) -> None:
        """Recomputed from the shipped node table: the closure is what the file claims it is."""
        profile = load_classifier_ontology()
        classes = profile["audioset"]["classes"]

        def closure(node_id: str, seen: set[str]) -> set[str]:
            if node_id in seen:
                return seen
            seen.add(node_id)
            for child in classes[node_id]["child_ids"]:
                if child in classes:
                    closure(child, seen)
            return seen

        for label, entry in profile["mapping"].items():
            expected: set[str] = set()
            for root in entry["audioset_ids"]:
                expected |= closure(root, set())
            assert set(entry["corroborating_ids"]) == expected, label

    def test_no_corroborating_class_is_outside_audiosets_released_527(self) -> None:
        """An ontology-only node is one no classifier has an output for."""
        for entry in load_classifier_ontology()["mapping"].values():
            assert entry["unreleased"] == []

    def test_a_profile_referencing_an_absent_id_is_refused(self, tmp_path: Path) -> None:
        """The guard fires rather than resolving the label to a silently short set."""
        profile: dict[str, Any] = json.loads((sorted(PROFILE_DIR.glob("*.json"))[-1]).read_text())
        profile["mapping"]["Cough"]["corroborating_ids"].append("/m/notaclass")
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(profile))
        with pytest.raises(ValueError, match="absent from the ontology"):
            load_classifier_ontology(str(path))

    def test_a_profile_of_an_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        """A reader that shrugged at a version bump would read fields that had moved."""
        profile: dict[str, Any] = json.loads((sorted(PROFILE_DIR.glob("*.json"))[-1]).read_text())
        profile["profile_version"] = "99"
        path = tmp_path / "future.json"
        path.write_text(json.dumps(profile))
        with pytest.raises(ValueError, match="profile_version"):
            load_classifier_ontology(str(path))

    def test_a_named_but_absent_profile_raises(self, tmp_path: Path) -> None:
        """A typo'd path is an operator error, not a reason to fall back to the bundled profile."""
        with pytest.raises(FileNotFoundError):
            load_classifier_ontology(str(tmp_path / "nothing.json"))


class TestTheLabelTuplesAreDerived:
    """The four routing tuples were the same duplication in a second place."""

    def test_the_hear_groups_come_from_the_profile(self) -> None:
        """A HeAR label joins a group in the profile, not in a tuple literal."""
        assert HEAR_COUGH == hear_labels_in_group("cough")
        assert HEAR_BREATH == hear_labels_in_group("breath")

    def test_the_audioset_tuples_are_the_groups_closures(self) -> None:
        """Each AudioSet tuple is the union of its group's corroboration sets, sorted."""
        assert AUDIOSET_COUGH == audioset_labels_for_group("cough")
        assert AUDIOSET_BREATH == audioset_labels_for_group("breath")

    def test_the_derived_tuples_hold_what_the_ontology_says(self) -> None:
        """Pinned so a regenerated profile that moves a class is visible rather than absorbed."""
        assert AUDIOSET_COUGH == ("Cough", "Sneeze", "Throat clearing")
        assert AUDIOSET_BREATH == ("Breathing", "Gasp", "Pant", "Snoring", "Snort", "Wheeze")
        assert set(HEAR_COUGH) == {"Cough", "Baby Cough", "Sneeze", "Throat Clear"}
        assert set(HEAR_BREATH) == {"Snore", "Breathe"}


class TestTheAirwayKindIsOneOntologyClosure:
    """The airway set was hand-listed in two files that disagreed; it is now the roots' closure."""

    def test_the_audioset_set_is_the_closure_of_the_configured_roots(self) -> None:
        """Recomputed from the shipped node table rather than compared against a second list."""
        profile = load_classifier_ontology()
        classes = profile["audioset"]["classes"]
        roots = [node_id for node_id, entry in classes.items() if entry["name"] in load_airway_roots()]
        assert roots, "the packaged roots must name AudioSet classes"
        reached: set[str] = set()
        stack = list(roots)
        while stack:
            node_id = stack.pop()
            if node_id in reached:
                continue
            reached.add(node_id)
            stack.extend(classes[node_id]["child_ids"])
        expected = sorted(
            classes[node_id]["name"]
            for node_id in reached
            if classes[node_id]["in_audioset_527"] or classes[node_id]["in_yamnet_521"]
        )
        assert list(AUDIOSET_AIRWAY) == expected

    def test_the_packaged_roots_are_the_respiratory_subtree(self) -> None:
        """Pinned, so widening the kind is a visible config change rather than a quiet one."""
        assert load_airway_roots() == ["Respiratory sounds"]
        assert AUDIOSET_AIRWAY == (
            "Breathing",
            "Cough",
            "Gasp",
            "Pant",
            "Sneeze",
            "Sniff",
            "Snoring",
            "Snort",
            "Throat clearing",
            "Wheeze",
        )

    def test_sigh_is_not_airway_evidence(self) -> None:
        """The hand list carried it; AudioSet places `Sigh` under `Human voice`, not respiration."""
        assert "Sigh" not in AUDIOSET_AIRWAY

    def test_pant_and_snort_are_airway_evidence(self) -> None:
        """Both are children of `Breathing` and both are in YAMNet's 521; the hand list omitted them."""
        assert {"Pant", "Snort"} <= set(AUDIOSET_AIRWAY)

    def test_sniff_is_airway_evidence(self) -> None:
        """`Sniff` is a sibling of `Breathing` under the root, so no HeAR-mapped subtree reaches it."""
        assert "Sniff" in AUDIOSET_AIRWAY
        assert "Sniff" not in set(AUDIOSET_COUGH) | set(AUDIOSET_BREATH)

    def test_the_two_hear_group_closures_are_not_the_airway_set(self) -> None:
        """The union answers a corroboration question about HeAR's labels, not what the kind is."""
        assert set(AUDIOSET_COUGH) | set(AUDIOSET_BREATH) < set(AUDIOSET_AIRWAY)

    def test_the_hear_labels_are_those_whose_mapped_node_is_inside_the_closure(self) -> None:
        """One root list drives both vocabularies, so the two cannot disagree."""
        assert HEAR_AIRWAY == hear_labels_under_roots(load_airway_roots())
        assert set(HEAR_AIRWAY) == {"Cough", "Snore", "Baby Cough", "Breathe", "Sneeze", "Throat Clear"}
        assert "Laugh" not in HEAR_AIRWAY and "Speech" not in HEAR_AIRWAY

    def test_a_node_no_classifier_can_emit_is_in_no_derived_set(self) -> None:
        """`Respiratory sounds` is in neither the 527 nor the 521, so it can never match."""
        classes = load_classifier_ontology()["audioset"]["classes"]
        unemittable = {
            entry["name"] for entry in classes.values() if not (entry["in_audioset_527"] or entry["in_yamnet_521"])
        }
        assert "Respiratory sounds" in unemittable
        derived = {
            name for group in (*FAMILIES.values(), *LABEL_SETS.values()) for names in group.values() for name in names
        }
        assert not derived & unemittable

    def test_every_derived_audioset_label_is_a_class_in_the_profile(self) -> None:
        """A name the profile does not carry is a name no classifier reports under that spelling."""
        names = {entry["name"] for entry in load_classifier_ontology()["audioset"]["classes"].values()}
        assert set(AUDIOSET_AIRWAY) <= names

    def test_every_derived_hear_label_is_one_the_profile_maps(self) -> None:
        """The HeAR side is drawn from the same profile, in the detector's own graph order."""
        assert set(HEAR_AIRWAY) <= set(hear_labels())
        assert list(HEAR_AIRWAY) == [label for label in hear_labels() if label in set(HEAR_AIRWAY)]

    def test_a_root_the_ontology_does_not_hold_is_refused(self) -> None:
        """A misspelled root would silently shrink the kind rather than fail."""
        with pytest.raises(ValueError, match="not AudioSet classes"):
            audioset_labels_under_roots(["Respiratory sound"])

    def test_naming_a_second_root_widens_the_kind(self, tmp_path: Path) -> None:
        """Restoring `Sigh` as airway evidence is one line of configuration, not a code change."""
        override = tmp_path / "roots.yaml"
        override.write_text("taxonomy:\n  airway_ontology_roots: [Respiratory sounds, Sigh]\n")
        config = load_triage_config(override)
        assert "Sigh" in airway_audioset_labels(config)
        assert set(AUDIOSET_AIRWAY) < set(airway_audioset_labels(config))
        assert airway_hear_labels(config) == HEAR_AIRWAY

    def test_the_packaged_configuration_is_what_the_routing_tuples_read(self) -> None:
        """The config key and the routing tuple are one definition, not two that agree today."""
        config = load_triage_config()
        assert airway_audioset_labels(config) == AUDIOSET_AIRWAY
        assert airway_hear_labels(config) == HEAR_AIRWAY


class TestLabelIdentity:
    """``canonical_names`` answers which ontology node a spelling denotes, not what corroborates it."""

    def test_every_audioset_class_names_itself(self) -> None:
        """An AudioSet display name is already the ontology's own name for its node."""
        identity = canonical_names()
        classes = load_classifier_ontology()["audioset"]["classes"]
        assert all(identity[entry["name"]] == entry["name"] for entry in classes.values())

    @pytest.mark.parametrize(
        ("spelling", "node"),
        [
            ("Throat Clear", "Throat clearing"),
            ("Snore", "Snoring"),
            ("Baby Cough", "Cough"),
            ("Breathe", "Breathing"),
            ("Laugh", "Laughter"),
            ("Cough", "Cough"),
            ("Sneeze", "Sneeze"),
            ("Speech", "Speech"),
        ],
    )
    def test_each_hear_label_names_its_mapped_node(self, spelling: str, node: str) -> None:
        """All eight, including the three whose spelling already matched."""
        assert canonical_names()[spelling] == node

    def test_identity_is_the_mapped_root_and_not_the_subtree(self) -> None:
        """``Throat clearing`` corroborates ``Cough`` and is still its own node."""
        identity = canonical_names()
        assert identity["Throat clearing"] == "Throat clearing"
        assert "Throat clearing" in corroboration_sets()["Cough"]

    def test_a_spelling_the_profile_does_not_hold_is_absent(self) -> None:
        """The map names what the ontology names; a caller decides what to do with the rest."""
        assert "Not an AudioSet class" not in canonical_names()

    def test_a_profile_whose_label_shadows_a_different_class_is_refused(self, tmp_path: Path) -> None:
        """One spelling standing for two nodes would merge two different events into one row."""
        profile: dict[str, Any] = json.loads((sorted(PROFILE_DIR.glob("*.json"))[-1]).read_text())
        sneeze = [node_id for node_id, entry in profile["audioset"]["classes"].items() if entry["name"] == "Sneeze"]
        profile["mapping"]["Cough"]["audioset_ids"] = sneeze
        path = tmp_path / "shadowed.json"
        path.write_text(json.dumps(profile))
        with pytest.raises(ValueError, match="itself an AudioSet class name"):
            load_classifier_ontology(str(path))
