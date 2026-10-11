# r31 replay errors: what caused the 29 errored rows, and what changed

The r31 replay from BACKGROUND over 62,550 stores errored on 29 rows: 27 in QUALITY, 1 in SPEECH and 1 in REPORT.

## QUALITY: `no stream named 'recording'` (27 rows)

All 27 stores are 2026-09-20 design-tree stores in which ADMIT failed with "every sample is zero". ADMIT wrote no
stream, PREPROCESS never ran, and the original graph pass skipped QUALITY. The replay runs QUALITY on every store, and
QUALITY raised because the `recording` stream it names in `used` was absent. No stream had been renamed: none was
ever written.

QUALITY reads no audio, so the stream only appears as a provenance `used` edge. When the stream is absent, the
report is now undetermined and carries `missing: ["recording"]` plus a `stream_not_stored` note. Clip spans without a
`clip_amplitude` measurement still raise.

## SPEECH: word ends 0.70 s past "the 22.016 s this audio decoded to" (1 row)

The word is a genuine recogniser span. Two ASR backends agree on its placement to within 0.04 s, and both put it
inside the 23.66 s recording. The decoded duration was wrong. The `plain.flac` sidecar is intact: libFLAC's decode
matches the STREAMINFO MD5 and the header's 378,579 frames. ffmpeg's FLAC parser, however, warns "sample/frame number
mismatch in adjacent frames" and stops at 352,256 samples (22.016 s). torchcodec goes through ffmpeg, so `Audio`
silently returned a recording that was 1.6 s short.

`bound_word_extent` still raises. Its refusal is what exposed a truncated decode. Clamping the word would have
hidden the missing audio from every node that reads the stream.

## REPORT: `decodeAVFrame ... Could not push packet to decoder` (1 row)

`preemphasised.flac` is intact by the same test: the libFLAC decode matches the MD5 over all 272,324 frames. ffmpeg
rejects one of its frames with "invalid residual". The file does not need PREPROCESS re-run.

## The shared fix

`Audio._lazy_load_data_from_filepath` falls back to libsndfile in two cases, for a file that libsndfile opens and
whose format carries an exact frame count (FLAC, WAV, W64, RF64, AIFF):

- torchcodec raises;
- a whole-file torchcodec decode returns fewer samples than the header's frame count.

Lossy formats keep torchcodec's result and its errors.
