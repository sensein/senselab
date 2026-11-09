"""The triage drivers: each one runs a single piece of triage over files, a corpus or a slice of one.

A driver is a module with ``main(argv) -> int``, its command line, and the functions that command calls (most have
``run_slice``, which takes ``--slice-index`` / ``--slice-count`` so a caller can split a corpus across jobs). Run one
from an installed senselab with ``python -m senselab.audio.workflows.triage.drivers.<name> ...``, or call ``main``
or its functions from Python. ``scripts/<name>.py`` in the repository runs the same module.

senselab provides the pieces; it does not fix the order a dataset runs them in. The caller, such as a dataset's own
pipeline, decides which pieces it runs, in what order and how it splits them into jobs.

One recording: ``triage_audio``.

Passes over finished runs (``extend_*``), each in place and resumable:
    ``extend_replay_decisions`` (re-decide from stored PREPROCESS output), ``extend_refold`` (decide again, reading
    nothing new), ``extend_background_floor`` and ``extend_session_floor`` (the session floor),
    ``extend_cohort`` (corpus quantiles, then speaker readings, then a consistency report),
    ``extend_llm_review`` and ``extend_second_opinion`` (the reviewer models), ``extend_task_audio`` (task-extent
    streams), ``extend_task_squim`` (SQUIM over the task extent), and the repair passes ``extend_clip_amplitudes``,
    ``extend_withdraw_clips``, ``extend_quality``, ``extend_diarization``, ``extend_ppg_praat`` and
    ``extend_reprocessed_outputs``.

Corpus outputs: ``triage_recording_vectors`` (one row per recording, in shards then merged), ``triage_table``
(``triage.tsv`` and its dictionary), ``triage_speaker_vectors``, ``triage_corpus_report``, ``triage_measure_stats``,
``triage_gate_breakdown``, ``triage_rerender``.

Review pages: ``triage_review_page``, ``free_speech_review_page``, ``triage_review_serve``,
``triage_review_cache_backfill``, ``triage_vectors_viewer``.

Profiles and checks: ``triage_profile_template``, ``fit_profile``, ``check_identity``, ``select_replay_manifest``,
``triage_replay_diff``, ``verify_derivatives_copy``, ``verify_task_squim``, ``build_classifier_ontology_map``.

Evaluation: ``evaluate_task_events``, ``score_taxonomy_ruleset``, ``analyze_routing_evidence``,
``gate_family_matrix``, ``measure_session_background``.
"""
