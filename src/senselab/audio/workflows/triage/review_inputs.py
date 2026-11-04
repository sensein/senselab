"""The redaction word lists and the reviewer's policy a run reads, from the active dataset profile (plan step 6b).

The word lists (``redaction_policy.yaml``: months, holidays, number and kinship words, countries, states) come with
senselab's reference profile ``adult_clinical_speech``, which a dataset may replace. The review policy
(``review_policy.txt``, the policy as the reviewer and the second opinion read it, and ``task_guidance.yaml``, what
counts as each task's content) is the dataset's own. The rules REDACT's fold applies stay senselab's until they
become a module (plan step 6b, option B).
"""

from __future__ import annotations

import hashlib

from senselab.audio.workflows.triage.dataset_profile import data_path, profile_cache
from senselab.text.tasks.pii_detection.redaction_policy import Lexicon, load_lexicon, version
from senselab.text.tasks.pii_detection.redaction_review import ReviewPolicy, load_review_policy, redaction_policy_text

LEXICON_FILE = "redaction_policy.yaml"
REVIEW_POLICY_FILE = "review_policy.txt"
TASK_GUIDANCE_FILE = "task_guidance.yaml"
"""The profile files these are read from."""


@profile_cache
def redaction_lexicon() -> Lexicon:
    """The active profile's redaction word lists.

    Returns:
        :func:`~senselab.text.tasks.pii_detection.redaction_policy.load_lexicon`'s table.

    Raises:
        DatasetDataMissing: If no active profile supplies :data:`LEXICON_FILE`.
    """
    return load_lexicon(data_path(LEXICON_FILE))


def lexicon_version() -> int:
    """The word lists' ``version``, recorded on every redaction ledger."""
    return version(redaction_lexicon())


@profile_cache
def review_policy() -> ReviewPolicy:
    """The active profile's review policy: its policy prose and its task guidance.

    Returns:
        The policy.

    Raises:
        DatasetDataMissing: If no active profile supplies :data:`REVIEW_POLICY_FILE` or :data:`TASK_GUIDANCE_FILE`.
    """
    return load_review_policy(data_path(REVIEW_POLICY_FILE), data_path(TASK_GUIDANCE_FILE))


@profile_cache
def policy_fingerprint() -> str:
    """The sha256 of what a reading is read under beyond its cache key: the policy text and the word lists.

    A cached reading records it, and a reading recorded under another one is read again (step-6 review finding 1).
    The cache key itself leaves them out, so a dataset's readings keep their keys while its policy is unchanged.

    Returns:
        The hex digest over the policy text as the reviewer is given it and the word-list file's bytes.
    """
    digest = hashlib.sha256(redaction_policy_text(review_policy()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(data_path(LEXICON_FILE).read_bytes())
    return digest.hexdigest()
