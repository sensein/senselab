"""The rule cascade's precision guards.

Every test here corresponds to a defect found in review of PR #542. They run on
the host against pure-Python helpers — the cascade's engine-dependent parts
(spaCy NER, gazetteer downloads) live in the subprocess worker and are covered
separately.
"""

import pytest

from senselab.text.tasks.pii_detection import rules


def test_zipf_returns_none_not_zero_without_wordfreq(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.0 means 'measured, maximally rare'.

    Every caller reads rarity as evidence FOR a PII hit. Returning 0.0 on a missing
    dependency inverts the precision guards instead of relaxing them.
    """
    monkeypatch.setattr(rules, "_WORDFREQ_IMPORT", None)
    assert rules._zipf("the") is None


def test_a_word_only_device_identifier_is_dropped() -> None:
    """An ASR'd cough token flagged as a 'device identifier' has no digits.

    Real MRNs, SSNs, and account numbers always do.
    """
    assert rules._valid_structured_identifier("cough", "IDNUM") is False
    assert rules._valid_structured_identifier("MRN 4417829", "IDNUM") is True


def test_contact_requires_an_at_sign_or_seven_digits_or_an_ip() -> None:
    """CONTACT format validity accepts an email sign, a 7+ digit run, or an IPv4 shape."""
    assert rules._valid_structured_identifier("jane@example.com", "CONTACT") is True
    assert rules._valid_structured_identifier("617 555 0134", "CONTACT") is True
    assert rules._valid_structured_identifier("192.168.1.10", "CONTACT") is True
    assert rules._valid_structured_identifier("telephone", "CONTACT") is False


def test_url_requires_a_url_shape() -> None:
    """URL format validity requires an actual URL shape, not just the word 'website'."""
    assert rules._valid_structured_identifier("https://example.com", "URL") is True
    assert rules._valid_structured_identifier("website", "URL") is False


def test_format_validation_is_not_switchable_by_recall_mode() -> None:
    """Format validity is a correctness check under either posture.

    Tying it to the precision flag let high-recall promote a word-only 'device identifier'
    straight to confirmed hard-gate PII — the opposite of what recall mode promises.
    """
    entities = [rules._entity(0, 5, "IDNUM", 0.99, "gliner")]
    for precision in (True, False):
        kept = rules.postprocess_entities(entities, "cough", precision_mode=precision)
        assert kept == [], f"word-only IDNUM survived with precision_mode={precision}"


def test_a_holiday_is_reclassified_as_a_date_not_a_name() -> None:
    """A NAME-tagged span that is actually a holiday name is reclassified to a DATE."""
    entities = [rules._entity(0, 9, "NAME", 0.9, "ner")]
    kept = rules.postprocess_entities(entities, "Christmas", precision_mode=True)
    assert kept and kept[0]["category"] == "DATE_PARTIAL"


def test_a_lone_common_word_name_is_not_hard_gate_eligible() -> None:
    """Will / May / Grant / Mark are the classic NER false positives.

    They drop to needs_review rather than failing a file.
    """
    assert (
        rules._name_hard_gate_eligible(
            span_text="Will",
            start=0,
            source_text="Will you read this",
            methods=set(),
            engines={"gliner"},
            score=0.7,
        )
        is False
    )


def test_a_multitoken_name_is_hard_gate_eligible() -> None:
    """A two-token span like 'Jane Doe' is hard-gate eligible without extra corroboration."""
    assert (
        rules._name_hard_gate_eligible(
            span_text="Jane Doe",
            start=0,
            source_text="Jane Doe speaking",
            methods=set(),
            engines={"gliner"},
            score=0.7,
        )
        is True
    )


def test_unknown_word_frequency_takes_the_precision_safe_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without wordfreq, 'is this a common word?' is unknown.

    Treating unknown as 'rare' would make every lone token hard-gate eligible.
    """
    monkeypatch.setattr(rules, "_zipf", lambda _word: None)
    assert (
        rules._name_hard_gate_eligible(
            span_text="Will",
            start=0,
            source_text="Will you read this",
            methods=set(),
            engines={"gliner"},
            score=0.7,
        )
        is False
    )


def test_rigidity_tiers_cover_every_weighted_category() -> None:
    """A category with a weight but no tier scores without being classified.

    That reads as a tier-less finding downstream.
    """
    for category in rules.CATEGORY_WEIGHTS:
        assert rules.rigidity_tier(category) != "misc" or category in rules.MISC


def test_age_over_ninety_is_flagged_and_under_is_not() -> None:
    """HIPAA Safe Harbor: ages over 89 are identifiers. Ages below are not."""
    assert rules.age_scan("I am ninety four years old", over_years=90)
    assert not rules.age_scan("I am forty two years old", over_years=90)


def test_gliner_window_offsets_are_absolute_not_window_relative() -> None:
    """A window's reported offset must index into the original text, not itself.

    GLiNER is run over overlapping word windows. A span's offsets come back relative to
    its window, so without re-basing they point at the wrong characters — and the masked
    preview then redacts the wrong text.
    """
    text = "one two three four five six seven eight nine ten Jane Doe"
    windows = list(rules._gliner_chunks(text, max_words=5, overlap_words=2))
    assert len(windows) > 1, "premise: this text must actually be split"
    for window_text, offset in windows:
        assert text[offset : offset + len(window_text)] == window_text


def test_gliner_windows_cover_every_word() -> None:
    """A word that falls in no window is a word GLiNER never sees.

    Truncation is the defect this windowing exists to remove, so losing the tail to an
    off-by-one in the stride would reintroduce it in a quieter form.
    """
    text = " ".join(f"w{i}" for i in range(37))
    covered = set()
    for window_text, offset in rules._gliner_chunks(text, max_words=5, overlap_words=2):
        covered.update(text[offset : offset + len(window_text)].split())
    assert covered == set(text.split())


def test_gliner_short_text_is_a_single_window() -> None:
    """Text under the limit must not be split — windowing it would only add duplicates."""
    text = "my name is Jane Doe"
    assert list(rules._gliner_chunks(text, max_words=50, overlap_words=10)) == [(text, 0)]


def test_flag_all_pii_takes_effect_independently_of_recall_mode() -> None:
    """The flag is a lever of its own, not a synonym for high recall.

    In #542 it was read only inside ``if HIGH_RECALL:``, so on its own it did nothing —
    silently. The two are separate levers: "report every candidate" is not the same
    request as "lower the detection thresholds".
    """
    entities = [rules._entity(0, 4, "AGE", 0.2, "age")]
    assert rules.postprocess_entities(entities, "I am 42", precision_mode=True, flag_all=True) == entities
    assert rules.postprocess_entities(entities, "I am 42", precision_mode=True, flag_all=False) == []


def test_flag_all_does_not_defeat_format_validation() -> None:
    """Reporting every candidate must not promote a known hallucination class.

    A word-only "device identifier" is wrong under either posture, so format validation
    runs regardless — the same reasoning that keeps it out of ``precision_mode``'s reach.
    """
    entities = [rules._entity(0, 6, "IDNUM", 0.9, "pattern")]
    assert rules.postprocess_entities(entities, "banana split", precision_mode=False, flag_all=True) == []
