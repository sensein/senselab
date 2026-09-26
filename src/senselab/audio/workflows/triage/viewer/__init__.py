"""The two browser views over ``recording_vectors.parquet``, and the build that inlines them.

The page is a single self-contained HTML file opened from ``file://``. It is assembled from the
parts in this directory by :func:`build`, and the assembled file is committed beside them so the
reader never needs a build step. ``viewer_test.py`` fails when the two disagree.

The page renders transcript text and detected PII extents, so it inherits the handling of the
parquet it reads: opened from disk by the owner, never hosted.
"""

from senselab.audio.workflows.triage.viewer.build import (
    BUILT_PAGE,
    INLINE_PATTERN,
    PARTS,
    SOURCE_DIR,
    build,
    write,
)

__all__ = ["BUILT_PAGE", "INLINE_PATTERN", "PARTS", "SOURCE_DIR", "build", "write"]
