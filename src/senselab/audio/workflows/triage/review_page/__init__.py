"""The triage review page: every recording's decision and evidence, faceted, with a reviewer's export.

Three tabs over one selection: Explore (parallel coordinates from the recording-vectors viewer),
Review (facets, search, and one recording's evidence, spectrogram and, for a speech task, its
transcript and redactions) and Decisions (the reviewer's entries, exported as JSON).
``specs/20261007-task-events-in-background/design.md`` ("Unit C plan", item 5) is the design.
"""

from senselab.audio.workflows.triage.review_page.page import index_of, page_html, write_page
from senselab.audio.workflows.triage.review_page.records import overlays, review_record, stream_paths
from senselab.audio.workflows.triage.review_page.spectrogram import pack, quantised_levels, unpack

__all__ = [
    "index_of",
    "overlays",
    "pack",
    "page_html",
    "quantised_levels",
    "review_record",
    "stream_paths",
    "unpack",
    "write_page",
]
