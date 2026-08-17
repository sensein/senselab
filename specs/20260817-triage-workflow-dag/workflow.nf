#!/usr/bin/env nextflow
//
// The triage workflow, as reviewable pseudocode.
//
// NOT RUNNABLE. This file exists so the dependency structure can be read as code rather than as
// prose: every task declares its input and output ports, every edge is a channel, and the round
// loop is a conditional re-entry of one sub-workflow rather than an unrolled copy of the graph.
// `ports.md` is the specification; where this file and that one disagree, that one wins.
//
// WHY NEXTFLOW DSL2 AND NOT CWL
// -----------------------------
// CWL expresses a static DAG well and conditionals badly. Its `when` clause skips a step on a
// per-invocation predicate, and iteration to a data-dependent fixed point has no standard
// construct at all — the usual CWL answer is to unroll the loop to a fixed depth, which is exactly
// what this design is trying not to do: an unrolled loop makes the round count a structural
// property of the file instead of a stopping decision with stated criteria. DSL2 has first-class
// workflow composition, so a sub-workflow is a task with ports, and it has a recursion construct
// whose exit predicate is written out as a boolean over the emitted state. That is the one thing
// the brief singled out, so it decided the notation.
//
// Two caveats stated rather than hidden: Nextflow's workflow recursion is a preview feature, and
// the real implementation is Python, not Nextflow. Neither affects the review — what is being
// reviewed is the port graph and the loop's exit conditions.
//
// CONVENTIONS
// -----------
//   * Parameters are passed explicitly as `val` inputs. No process reads a global. `cfg` below is
//     shorthand for "this key of the one versioned config"; the real key names are in ports.md §3.
//   * `emit:` names are the output port names from ports.md §2. One producer per name.
//   * A process whose input channel is empty does not execute and emits nothing. That is the only
//     skip mechanism in this design — there is no skip flag, and `stages`-style booleans do not
//     appear.
//   * `cache: true` marks the processes that hit the content-addressable cache. The key is
//     (waveform signature, task, model id, resolved commit sha, canonical params, wrapper code
//     version, senselab version, schema version) — cached_inference.py:383-394, schema 23 at :64.

nextflow.preview.recursion = true

// ============================================================================================
// ADMIT — is there anything here, and which audio variants exist
// ============================================================================================

process decode_audio {
    cache true
    input:
        path  audio_file
        val   device
    output:
        path  "audio_raw.wav",       emit: audio_raw
        val   audio_signature,       emit: audio_signature
    script:
        // exists: senselab.audio.tasks.preprocessing (resample_audios, downmix_audios_to_mono);
        //         audio_signature at utils/tasks/cached_inference.py:319
        "true"
}

process level_and_floor {
    input:
        path  audio_raw
        val   floor_percentile        // cfg.quality.floor_percentile  -- key exists, unread today
        val   gain_cap_db             // cfg.quality.max_gain_cap_db   -- key exists, unread today
    output:
        path  "level_track.parquet",  emit: level_track
        path  "band_floor.parquet",   emit: band_floor
        path  "clip_track.parquet",   emit: clip_track
    script:
        // exists but unwired: level.py:138-238 (integrated_lufs, true_peak_dbtp, clipped_fraction)
        //   -- reachable only through measure_variant (level.py:292), which has no caller
        // exists and wired: noise_floor.estimate_noise_floor (noise_floor.py:257)
        "true"
}

process signal_gate {
    input:
        path  level_track
        path  band_floor
        val   speech_threshold        // cfg.triage.speech_threshold
        val   min_speech_s            // cfg.triage.min_speech_s
    output:
        path  "signal_present.json",  emit: signal_present   // an Estimate
        path  "usable.flag",          emit: usable, optional: true
    script:
        // partial: adaptive/triage.py:21-155 (triage_decision) is the nearest thing, called once
        //   from scripts/analyze_audio.py:559. It decides "was there speech", not "is there signal".
        "true"
}

process enhance_audio {
    cache true
    input:
        path  audio_raw
        val   enhancement_model       // cfg.models.enhancement
        val   mode                    // cfg.enhancement.mode
        val   snr_floor_db            // cfg.triage.snr_floor_db      -- declared underived in the YAML
        val   low_snr_fraction        // cfg.triage.low_snr_fraction
    output:
        path  "audio_enhanced.wav",   emit: audio_enhanced
    script:
        // exists: senselab.audio.tasks.speech_enhancement.enhance_audios (api.py:19)
        "true"
}

workflow ADMIT {
    take:
        ch_audio_file
        cfg_device
        cfg_enhancement_model
        cfg_enhancement_mode
        cfg_snr_floor_db
        cfg_low_snr_fraction
        cfg_floor_percentile
        cfg_gain_cap_db
    main:
        decode_audio(ch_audio_file, cfg_device)
        level_and_floor(decode_audio.out.audio_raw, cfg_floor_percentile, cfg_gain_cap_db)
        signal_gate(level_and_floor.out.level_track, level_and_floor.out.band_floor,
                    cfg_device /* placeholder for cfg.triage.speech_threshold */, cfg_gain_cap_db)

        // The gate filters the stream. Downstream tasks joined against `usable` receive nothing
        // when the file is dead, so they do not run and produce no output.
        ch_usable = decode_audio.out.audio_raw.join(signal_gate.out.usable).map { a, _f -> a }

        enhance_audio(ch_usable, cfg_enhancement_model, cfg_enhancement_mode,
                      cfg_snr_floor_db, cfg_low_snr_fraction)
    emit:
        audio_raw       = ch_usable
        audio_enhanced  = enhance_audio.out.audio_enhanced
        audio_signature = decode_audio.out.audio_signature
        level_track     = level_and_floor.out.level_track
        band_floor      = level_and_floor.out.band_floor
        clip_track      = level_and_floor.out.clip_track
        signal_present  = signal_gate.out.signal_present
}

// ============================================================================================
// TAXONOMY — what kinds of sound are in here. The root answer.
// ============================================================================================

process speech_frame_posterior {
    cache true
    input:
        path  audio
        val   device
    output:
        path  "speech_posterior.parquet", emit: speech_posterior
    script:
        // exists: voice_activity_detection/frame_posteriors.py -- chunked_frame_inference at :241,
        //   FramePosterior at :108. Reaches speech_presence.harvest_speech_presence_evidence
        //   (speech_presence.py:148) through an optional `frame_posteriors=` argument.
        "true"
}

process sound_event_posterior {
    cache true
    input:
        path  audio
        val   ast_model               // cfg.models.ast
        val   yamnet_model            // cfg.models.yamnet
        val   win_length              // cfg.scene.ast_win_length / yamnet_win_length
        val   hop_length              // cfg.scene.ast_hop_length / yamnet_hop_length
        val   top_k                   // cfg.scene.top_k
    output:
        path  "event_posterior.parquet", emit: event_posterior
    script:
        // partial: classification/api.py:32 (classify_audios, windowed) gives 0.96 s windows at a
        //   0.48 s hop for both AST and YAMNet -- window-level, not frame-level. A frame-level
        //   AudioSet-Strong model is the gap; see design.md section 3.
        "true"
}

process voicing_track {
    cache true
    input:
        path  audio
        val   win_length              // cfg.features.win_length
        val   hop_length              // cfg.features.hop_length
    output:
        path  "voicing_track.parquet", emit: voicing_track
    script:
        // exists: features_extraction/opensmile.py:74 (10 ms LLDs incl. HNR),
        //         features_extraction/ppg.py:136 (voice fraction, subprocess venv "ppgs")
        "true"
}

process taxonomy_fold {
    input:
        path  speech_posterior
        path  event_posterior
        path  voicing_track
        path  level_track
        path  band_floor
        val   grid_win_length         // cfg.grid.win_length
        val   grid_hop_length         // cfg.grid.hop_length
        val   speech_presence_labels  // cfg.uncertainty.speech_presence_labels
        val   linking                 // cfg.linking.{frame_speech_threshold, label_mass_threshold,
                                      //   speech_excess_db, lufs_silence, lufs_speech,
                                      //   coarse_voter_weight, coarse_window_ratio}
        val   taxonomy_classes        // cfg.taxonomy.classes        -- ABSENT, no such section
        val   voter_weights           // cfg.taxonomy.voter_weights  -- ABSENT
        val   aggregator              // cfg.uncertainty.aggregator
    output:
        path  "taxonomy_track.parquet", emit: taxonomy_track
        path  "vocal_spans.parquet",    emit: vocal_spans
    script:
        // does not exist as a four-way fold. The nearest code is
        //   speech_presence.harvest_speech_presence_evidence (speech_presence.py:148) plus
        //   speech_presence_link.link_speech_presence (speech_presence_link.py:497), which produce a
        //   binary speech-present belief, and sound_sources.harvest_source_categories
        //   (sound_sources.py:175), which produces four *source* categories -- speech, people,
        //   machine, environment -- where a cry lands in `people`, a background category. That is
        //   the substitution this task exists to stop making.
        "true"
}

process content_gate {
    input:
        path  taxonomy_track
        val   evidence                // cfg.evidence.content.{prior, pseudo_count, floor, population}
                                      //   -- ABSENT, no such section
    output:
        path  "content.json",         emit: content
    script:
        // does not exist. Estimate itself does: utils/data_structures/estimate.py:28, with
        //   value/raw/n_evidence/prior and Estimate.no_evidence() at :137 -- and zero consumers
        //   anywhere in audio_analysis.
        "true"
}

workflow TAXONOMY {
    take:
        ch_audio_raw
        ch_level_track
        ch_band_floor
        cfg_device
        cfg_ast_model
        cfg_yamnet_model
        cfg_scene
        cfg_features
        cfg_grid
        cfg_linking
        cfg_labels
        cfg_taxonomy
        cfg_aggregator
        cfg_evidence
    main:
        speech_frame_posterior(ch_audio_raw, cfg_device)
        sound_event_posterior(ch_audio_raw, cfg_ast_model, cfg_yamnet_model,
                              cfg_scene, cfg_scene, cfg_scene)
        voicing_track(ch_audio_raw, cfg_features, cfg_features)
        taxonomy_fold(speech_frame_posterior.out.speech_posterior,
                      sound_event_posterior.out.event_posterior,
                      voicing_track.out.voicing_track,
                      ch_level_track, ch_band_floor,
                      cfg_grid, cfg_grid, cfg_labels, cfg_linking,
                      cfg_taxonomy, cfg_taxonomy, cfg_aggregator)
        content_gate(taxonomy_fold.out.taxonomy_track, cfg_evidence)
    emit:
        taxonomy_track  = taxonomy_fold.out.taxonomy_track
        vocal_spans     = taxonomy_fold.out.vocal_spans
        event_posterior = sound_event_posterior.out.event_posterior
        content         = content_gate.out.content
}

// ============================================================================================
// SPEECH CONTENT — what was said, whether it matches the task, whether it identifies anyone
// ============================================================================================

process transcribe {
    cache true
    input:
        path  audio                   // one channel item per (variant, model)
        val   model_id                // one of cfg.models.asr
        val   biasing_vocabulary      // hints.expected_speech, or empty
    output:
        path  "hypotheses.json",      emit: hypotheses
    script:
        // exists: speech_to_text/api.py:55 (transcribe_audios), wired through
        //   stages.stage_asr (stages.py:323) behind run_task_cached.
        // NOTE: adaptive/backends.py:93 calls the same transcribe_audios with no cache and no
        //   provenance -- a second, unrecorded invocation surface. See design.md section 7.
        "true"
}

process align_words {
    cache true
    input:
        path  audio
        path  hypotheses
        val   aligner                 // cfg.alignment.aligner
        val   qwen_model              // cfg.alignment.qwen_model
        val   mms_model               // cfg.alignment.mms_model
        val   language                // cfg.alignment.language
        val   native_timestamps       // cfg.alignment.qwen_native_timestamps
    output:
        path  "word_times.parquet",   emit: word_times
    script:
        // exists: stages.stage_alignment (stages.py:363) behind run_alignment_cached; the key is
        //   separable from ASR's -- align_cache_key at cached_inference.py:398.
        "true"
}

process fuse_words {
    input:
        path  word_times              // collected across every model
        val   slot_overlap            // cfg.linking.asr_slot_overlap
        val   slot_mid_tol_s          // cfg.linking.asr_slot_mid_tol_s
    output:
        path  "transcript.json",      emit: transcript
    script:
        // exists: asr.fuse_consensus_words (asr.py:194) over
        //   speech_to_text_ensemble.fuse_word_streams (api.py:197).
        // The `policy=` argument carrying the two slot parameters was dropped at the only reachable
        //   call site until the F-162 fix; the parameters above are what that fix restored.
        "true"
}

process transcript_gate {
    input:
        path  transcript
        path  vocal_spans
        val   evidence                // cfg.evidence.transcript.*  -- ABSENT
    output:
        path  "transcript_confidence.json", emit: transcript_confidence
    script:
        // does not exist. A voiced span with no words is currently read as silence rather than as a
        //   measured disagreement -- the assumption behind F-165.
        "true"
}

process task_match_gate {
    input:
        path  expected_speech         // hints.expected_speech -- NO PRODUCER TODAY
        path  transcript
        path  taxonomy_track
        val   task_type               // cfg.task.type
    output:
        path  "task_match.json",      emit: task_match
    script:
        // does not exist. AudioHints/ExpectedSpeech are declared (audio_hints.py:31,129) and read
        //   by nothing: grep for `.hints` over audio/workflows/ and scripts/analyze_audio.py finds
        //   zero hits, and audio_hints.py:5 says so outright.
        "true"
}

process pii_scan {
    cache true
    input:
        path  transcript
        val   detector                // rules | presidio | gliner
        val   presidio_score_threshold // cfg.pii.presidio_score_threshold -- ABSENT; literal pii.py:82
        val   gliner_threshold        // cfg.pii.gliner_threshold          -- ABSENT; literal pii.py:85
    output:
        path  "pii_candidates.json",  emit: pii_candidates
    script:
        // exists: text/tasks/pii_detection -- scan_for_pii (api.py:369), subprocess venv
        //   "pii-detection" (subprocess_backend.py:74). The scan/decide split already exists there.
        "true"
}

process pii_gate {
    input:
        path  pii_candidates          // collected across detectors
        path  word_times
        val   corroboration_min       // cfg.pii.corroboration_min -- ABSENT; literal `count >= 2` pii.py:241
        val   evidence                // cfg.evidence.pii.*        -- ABSENT
    output:
        path  "pii.json",             emit: pii
        path  "pii_spans.parquet",    emit: pii_spans
    script:
        // partial: text/tasks/pii_detection.decide_pii (api.py:515) and
        //   audio_analysis/pii.py:241 already do the aggregation; what does not exist is publishing
        //   the witness count as evidence rather than collapsing it into a boolean.
        "true"
}

workflow SPEECH_CONTENT {
    take:
        ch_audio_raw
        ch_audio_enhanced
        ch_vocal_spans
        ch_taxonomy_track
        ch_expected_speech            // may be empty; task_match_gate then does not run
        cfg_asr_models
        cfg_alignment
        cfg_linking
        cfg_task_type
        cfg_pii
        cfg_evidence
    main:
        ch_variants = ch_audio_raw.mix(ch_audio_enhanced)
        transcribe(ch_variants, cfg_asr_models, ch_expected_speech.ifEmpty([]))
        align_words(ch_variants, transcribe.out.hypotheses,
                    cfg_alignment, cfg_alignment, cfg_alignment, cfg_alignment, cfg_alignment)
        fuse_words(align_words.out.word_times.collect(), cfg_linking, cfg_linking)
        transcript_gate(fuse_words.out.transcript, ch_vocal_spans, cfg_evidence)
        task_match_gate(ch_expected_speech, fuse_words.out.transcript, ch_taxonomy_track, cfg_task_type)
        pii_scan(fuse_words.out.transcript, cfg_pii, cfg_pii, cfg_pii)
        pii_gate(pii_scan.out.pii_candidates.collect(), align_words.out.word_times,
                 cfg_pii, cfg_evidence)
    emit:
        transcript             = fuse_words.out.transcript
        word_times             = align_words.out.word_times
        transcript_confidence  = transcript_gate.out.transcript_confidence
        task_match             = task_match_gate.out.task_match      // absent if no expected task
        pii                    = pii_gate.out.pii
        pii_spans              = pii_gate.out.pii_spans
}

// ============================================================================================
// VOICE IDENTITY — how many voices, and is any of them not the target. RAW AUDIO ONLY.
// ============================================================================================

process window_embeddings {
    cache true
    input:
        path  audio_raw
        path  vocal_spans
        val   model_id                // one of cfg.models.embeddings
        val   window_s                // cfg.embeddings.window_s -- 0.5 in the config,
        val   hop_s                   // cfg.embeddings.hop_s    -- 0.25 in the config;
                                      //   compute.py:101-102 defaults to 2.0/1.0, so a caller that
                                      //   bypasses the config silently gets a 4x wider window
    output:
        path  "window_embeddings.parquet", emit: window_embeddings
    script:
        // exists: speaker_embeddings/windowing.py:51-88 (extract_per_window_embeddings)
        // CHANGED: windows are drawn on vocal_spans. Today the equivalent mask is
        //   compute._speech_window_mask (compute.py:893), a YAMNet top-1 veto whose own docstring
        //   names the child-voice-as-Music failure mode (F-170).
        "true"
}

process cluster_windows {
    input:
        path  window_embeddings
        val   cluster_cosine_threshold // cfg.speaker.cluster_cosine_threshold
        val   same_floor               // cfg.speaker.same_floor
        val   diff_floor               // cfg.speaker.diff_floor
        val   algorithm                // cfg.speaker.clustering_algorithm
    output:
        path  "cluster_structure.json", emit: cluster_structure
    script:
        // exists, and is DEAD IN PRODUCTION: embeddings.cluster_pass_speakers (embeddings.py:97) is
        //   reachable only from speech_presence_link.derive_window_clusters (:377), which nothing
        //   calls. Also note the function takes a mutable `failures` dict and writes to it
        //   (embeddings.py:162,190,244,254,299) -- an out-parameter, not a pure return.
        "true"
}

process diarize {
    cache true
    input:
        path  audio_raw
        val   model_id                // one of cfg.models.diarization
    output:
        path  "diarization.json",     emit: diarization
    script:
        // exists: speaker_diarization/api.py:65 via stages.stage_diarization (stages.py:136)
        "true"
}

process harmonize_labels {
    input:
        path  diarization             // collected across diarizers
        path  window_embeddings
        val   centroid_min_similarity // cfg.speaker.centroid_min_similarity
    output:
        path  "harmonized_speakers.json", emit: harmonized_speakers
        path  "overlap_track.parquet",    emit: overlap_track
    script:
        // exists: harmonize.harmonize_from_diarization (harmonize.py:307);
        //   overlap from occupancy.spans_from_diarization (occupancy.py:87) +
        //   adaptive/backends.overlap_track_from_spans (backends.py:237)
        "true"
}

process speaker_count_gate {
    input:
        path  cluster_structure
        path  harmonized_speakers
        val   supported_threshold     // cfg.speaker.supported_threshold
        val   evidence                // cfg.evidence.speaker_count.* -- ABSENT
    output:
        path  "speaker_count.json",   emit: speaker_count
    script:
        // partial: speaker_identity.speaker_count_posterior (speaker_identity.py:117) publishes a
        //   vote share as a posterior whose width does not shrink with more agreeing sources
        //   (F-179), and multimodal_threshold=0.15 flips the verdict between 5 and 6 sources with
        //   no change in the audio (F-144). Both are what the Estimate contract replaces.
        "true"
}

process novelty_track {
    input:
        path  window_embeddings
        path  cluster_structure
    output:
        path  "novelty_track.parquet", emit: novelty_track
    script:
        // does not exist. Nearest primitives: adaptive/identity_repair.change_point_trajectory
        //   (:53) and _agglomerative_cosine (:125); embeddings.calibrate_cosine_uncertainty (:611).
        "true"
}

process off_target_fold {
    input:
        path  novelty_track
        path  overlap_track
        path  event_posterior         // the babble / crowd / chatter subtree
        path  taxonomy_track
        path  transcript              // optional: read-passage deviation
        val   aggregator              // cfg.uncertainty.aggregator
    output:
        path  "off_target_track.parquet", emit: off_target_track
    script:
        // does not exist. `off_target` appears once in src/, in a docstring
        //   (source_separation/unasdiff.py:9). No field, column, function or artifact.
        "true"
}

process off_target_gate {
    input:
        path  off_target_track
        path  cluster_structure       // for dominant_share
        path  target_voice            // hints.target_speaker, or empty
        val   off_target_cfg          // cfg.off_target.* -- ABSENT
        val   evidence                // cfg.evidence.off_target.* -- ABSENT
    output:
        path  "off_target.json",         emit: off_target
        path  "off_target_spans.parquet", emit: off_target_spans
    script:
        // does not exist. With no target_voice the published claim is "a voice other than the
        //   majority voice is present"; with one it becomes verification. The distinction is the
        //   wire, not a caveat.
        "true"
}

workflow VOICE_IDENTITY {
    take:
        ch_audio_raw                  // NOTE: no audio_enhanced port. By construction.
        ch_vocal_spans
        ch_taxonomy_track
        ch_event_posterior
        ch_transcript
        ch_target_voice
        cfg_diarization_models
        cfg_embeddings_models
        cfg_embeddings
        cfg_speaker
        cfg_aggregator
        cfg_off_target
        cfg_evidence
    main:
        window_embeddings(ch_audio_raw, ch_vocal_spans, cfg_embeddings_models,
                          cfg_embeddings, cfg_embeddings)
        cluster_windows(window_embeddings.out.window_embeddings,
                        cfg_speaker, cfg_speaker, cfg_speaker, cfg_speaker)
        diarize(ch_audio_raw, cfg_diarization_models)
        harmonize_labels(diarize.out.diarization.collect(),
                         window_embeddings.out.window_embeddings, cfg_speaker)
        speaker_count_gate(cluster_windows.out.cluster_structure,
                           harmonize_labels.out.harmonized_speakers, cfg_speaker, cfg_evidence)
        novelty_track(window_embeddings.out.window_embeddings,
                      cluster_windows.out.cluster_structure)
        off_target_fold(novelty_track.out.novelty_track,
                        harmonize_labels.out.overlap_track,
                        ch_event_posterior, ch_taxonomy_track, ch_transcript, cfg_aggregator)
        off_target_gate(off_target_fold.out.off_target_track,
                        cluster_windows.out.cluster_structure,
                        ch_target_voice.ifEmpty([]), cfg_off_target, cfg_evidence)
    emit:
        speaker_count     = speaker_count_gate.out.speaker_count
        off_target        = off_target_gate.out.off_target
        off_target_spans  = off_target_gate.out.off_target_spans
        cluster_structure = cluster_windows.out.cluster_structure
        overlap_track     = harmonize_labels.out.overlap_track
}

// ============================================================================================
// QUALITY and TRIM
// ============================================================================================

process scene_quality_frames {
    cache true
    input:
        path  audio                   // both variants
        val   analysis_win_length     // cfg.quality.analysis_win_length -- key exists, unread today
        val   analysis_hop_length     // cfg.quality.analysis_hop_length -- key exists, unread today
    output:
        path  "snr_track.parquet",       emit: snr_track
        path  "c50_track.parquet",       emit: c50_track
        path  "bandwidth_track.parquet", emit: bandwidth_track
    script:
        // exists: scene_quality/brouhaha.py:220 (extract_brouhaha_frames), subprocess venv
        //   "brouhaha". Required-when-enabled: an unreachable Brouhaha fails the run loudly.
        "true"
}

process quality_measures {
    input:
        path  snr_track
        path  c50_track
        path  bandwidth_track
        path  level_track
        path  band_floor
        path  clip_track
        val   grid_win_length         // cfg.grid.win_length
        val   grid_hop_length         // cfg.grid.hop_length
    output:
        path  "quality_measures.parquet", emit: quality_measures
    script:
        // exists: quality.harvest_quality_measurements (quality.py:243) -- already emits dB / Hz /
        //   proportion with per-signal provenance and no scores. The cleanest existing L1 task.
        "true"
}

process degradation_gate {
    input:
        path  quality_measures
        path  content                 // selects the anchors -- this edge does not exist today
        val   calibration_profile     // cfg.profiles.calibration
        val   detection_margin_profile // cfg.profiles.detection_margin
    output:
        path  "quality.json",         emit: quality
    script:
        // exists: degradation.scene_degradation (degradation.py:129) -- but it takes no task-type or
        //   content input at all and its anchors are fixed at 25 dB SNR / 30 dB C50
        //   (degradation.py:33-44). That is F-169, and the `content` port is the fix.
        "true"
}

process defect_spans {
    input:
        path  quality_measures
    output:
        path  "defect_spans.parquet", emit: defect_spans
    script:
        // partial: clipped_fraction (level.py:209) exists but is only reachable through
        //   measure_variant (level.py:292), which has no caller.
        "true"
}

process trim_proposal {
    input:
        path  taxonomy_track
        path  off_target_spans
        path  pii_spans
        path  defect_spans
    output:
        path  "trim_candidates.parquet", emit: trim_candidates
    script:
        // does not exist. grep for trim_regions / propose_trim over src/ returns nothing; the only
        //   `trim` identifier in the package is background_mask.guard_trimmed_s, unrelated.
        // The prior design sourced trim from background_mask's `target_free` spans only -- a product
        //   that reaches speaker attribution through a port with no producer (F-187), and whose
        //   sibling state fires on YAMNet's `Silence` label. See design.md section 7.
        "true"
}

process trim_gate {
    input:
        path  trim_candidates
        val   evidence                // cfg.evidence.trim.* -- ABSENT
    output:
        path  "trim_regions.parquet", emit: trim_regions
    script:
        "true"
}

workflow QUALITY {
    take:
        ch_audio_raw
        ch_audio_enhanced
        ch_level_track
        ch_band_floor
        ch_clip_track
        ch_content
        cfg_quality
        cfg_grid
        cfg_profiles
    main:
        scene_quality_frames(ch_audio_raw.mix(ch_audio_enhanced), cfg_quality, cfg_quality)
        quality_measures(scene_quality_frames.out.snr_track,
                         scene_quality_frames.out.c50_track,
                         scene_quality_frames.out.bandwidth_track,
                         ch_level_track, ch_band_floor, ch_clip_track, cfg_grid, cfg_grid)
        degradation_gate(quality_measures.out.quality_measures, ch_content,
                         cfg_profiles, cfg_profiles)
        defect_spans(quality_measures.out.quality_measures)
    emit:
        quality          = degradation_gate.out.quality
        quality_measures = quality_measures.out.quality_measures
        defect_spans     = defect_spans.out.defect_spans
}

workflow TRIM {
    take:
        ch_taxonomy_track
        ch_off_target_spans
        ch_pii_spans
        ch_defect_spans
        cfg_evidence
    main:
        trim_proposal(ch_taxonomy_track, ch_off_target_spans, ch_pii_spans, ch_defect_spans)
        trim_gate(trim_proposal.out.trim_candidates, cfg_evidence)
    emit:
        trim_regions = trim_gate.out.trim_regions
}

// ============================================================================================
// DECIDE — the ledger and the flag
// ============================================================================================

process evidence_ledger {
    input:
        path  estimates               // every Estimate, collected
        path  spans                   // every span set, collected
        val   evidence                // cfg.evidence.* per answer -- ABSENT
    output:
        path  "ledger.json",          emit: ledger
    script:
        // does not exist. Estimate does: utils/data_structures/estimate.py:28. Nothing in
        //   audio_analysis constructs one; the workflow's `estimates.py` is a parquet column
        //   schema (estimates.py:101), not a type.
        "true"
}

process review_flag_gate {
    input:
        path  ledger
        val   low_threshold           // cfg.labelstudio.low_threshold  -- key exists, unread today
        val   high_threshold          // cfg.labelstudio.high_threshold -- key exists, unread today
    output:
        path  "review_flag.json",     emit: review_flag
        path  "reasons.json",          emit: reasons
    script:
        // Three arms, each naming itself when it fires:
        //   1. an answer crosses its review band AND has evidence at or above its floor
        //   2. an answer's evidence is below its floor            <- also the refinement trigger
        //   3. the recording contradicts a supplied expected task
        // A flag with no reasons is a bug. Arm 2 is the direct fix for a total-harvest-failure run
        //   publishing high_uncertainty_rate = 0.0 (F-150, now None with no denominator).
        "true"
}

workflow DECIDE {
    take:
        ch_estimates
        ch_spans
        cfg_labelstudio
        cfg_evidence
    main:
        evidence_ledger(ch_estimates.collect(), ch_spans.collect(), cfg_evidence)
        review_flag_gate(evidence_ledger.out.ledger, cfg_labelstudio, cfg_labelstudio)
    emit:
        ledger      = evidence_ledger.out.ledger
        review_flag = review_flag_gate.out.review_flag
        reasons     = review_flag_gate.out.reasons
}

// ============================================================================================
// REFINE — the round loop. One sub-workflow, re-entered conditionally.
// ============================================================================================

process rank_undecided {
    input:
        path  ledger
        val   action_history          // every action set already executed
        val   epistemic_tolerance     // cfg.rounds.epistemic_tolerance -- key exists, unread today
        val   cycle_window            // cfg.rounds.cycle_window        -- key exists, unread today
    output:
        path  "candidate_actions.json", emit: candidate_actions
    script:
        // partial: adaptive/regions.propose_regions (regions.py:11) proposes where, and
        //   adaptive/interventions.RULES (interventions.py:1142) proposes what. Four of the nine
        //   rules -- S1, I1, I2, I4 -- have triggers containing no uncertainty term at all.
        "true"
}

process stop_or_continue {
    input:
        path  candidate_actions
        val   budget_remaining
        val   max_rounds              // cfg.rounds.max_rounds
    output:
        path  "stop_reason.json",     emit: stop_reason, optional: true
        path  "actions.json",         emit: actions,     optional: true
    script:
        // EXIT CRITERIA, stated once and in one place:
        //   DECIDED     -- every answer's Estimate is outside its ambiguity band with evidence at
        //                  or above its floor
        //   IRREDUCIBLE -- an answer is still ambiguous, and no unused action is predicted to add
        //                  an independent source to it
        //   OSCILLATING -- the planned action set repeats one already in action_history
        //   EXHAUSTED   -- budget_remaining is spent, or the round index reaches max_rounds
        // Exactly one of stop_reason and actions is emitted. stop_reason is published as an output:
        //   IRREDUCIBLE plus an ambiguous answer is the honest terminal state, and it is what
        //   routes the file to a human.
        //
        // Today there are TWO loops with different exit criteria in one process:
        //   fuse.py:1065 uses rounds.assess_convergence (C1-C4 at rounds.py:322-328), and
        //   adaptive/loop.py:224 exits at loop.py:315-317 where "converged" means only that no
        //   intervention fired -- assess_convergence is never called from adaptive/. This design
        //   has one loop and one criterion set.
        "true"
}

process narrow_input {
    input:
        path  actions
        path  audio_raw
    output:
        path  "region_*.wav",         emit: audio_regions
    script:
        // A narrowed region is materialised as audio, so it gets its own waveform signature and its
        //   own cache entry -- no new cache key field is needed. Offsets must be restored on the way
        //   out. adaptive/backends.py already crops this way, but bypasses run_task_cached.
        "true"
}

workflow REFINE {
    take:
        ch_ledger                     // ledger at round k
        ch_budget                     // budget_remaining at round k
        ch_history                    // action sets already executed
        ch_audio_raw
        cfg_rounds
    main:
        rank_undecided(ch_ledger, ch_history, cfg_rounds, cfg_rounds)
        stop_or_continue(rank_undecided.out.candidate_actions, ch_budget, cfg_rounds)
        narrow_input(stop_or_continue.out.actions, ch_audio_raw)

        // Re-entry: the same sub-workflows, over the narrowed audio. Every port is the same port;
        // only the audio and the ledger differ. This is the whole of "the rounds are a conditional
        // re-entry of the same workflow".
        ch_next = MEASURE_AND_DECIDE(narrow_input.out.audio_regions, ch_ledger)
    emit:
        ledger  = ch_next.ledger      // ledger at round k+1 -- a new, distinct value
        budget  = ch_next.budget
        history = ch_next.history
        stop    = stop_or_continue.out.stop_reason.ifEmpty([])
}

// The measurement half, factored out so both round 0 and every later round call the same thing.
workflow MEASURE_AND_DECIDE {
    take:
        ch_audio
        ch_prior_ledger               // empty at round 0
    main:
        // TAXONOMY -> SPEECH_CONTENT, VOICE_IDENTITY, QUALITY -> TRIM -> DECIDE, exactly as wired
        // above. Elided here rather than repeated: the point of this block is that it is one
        // callable unit with ports, so the loop has something to re-enter.
        Channel.empty().set { ledger }
    emit:
        ledger  = ledger
        budget  = Channel.empty()
        history = Channel.empty()
}

// ============================================================================================
// The root workflow
// ============================================================================================

workflow {
    ch_audio_file = Channel.fromPath(params.audio)
    ch_hints      = params.hints ? Channel.fromPath(params.hints) : Channel.empty()

    ADMIT(ch_audio_file, params.cfg.device, params.cfg.models.enhancement,
          params.cfg.enhancement.mode, params.cfg.triage.snr_floor_db,
          params.cfg.triage.low_snr_fraction, params.cfg.quality.floor_percentile,
          params.cfg.quality.max_gain_cap_db)

    round0 = MEASURE_AND_DECIDE(ADMIT.out.audio_raw, Channel.empty())

    // The conditional round loop. `until` is the exit predicate, and it is satisfied exactly when
    // stop_or_continue emitted a stop_reason -- which it does for one of the four stated reasons and
    // for no other.
    REFINE.recurse(round0.ledger, round0.budget, round0.history, ADMIT.out.audio_raw,
                   params.cfg.rounds)
          .until { ledger, budget, history, stop -> stop.size() > 0 }
}
