# T20R-10 evidence: scale ladder at one SHA (3262a713), measured on one H100 cluster and one CPU class

Evidence for the per-resource rules and for each group's own check. Nothing here is a packaged default (owner rulings 2026-11-09, R-252).
Jobs: GPU suite entries 0-4 in one job and entry 5 in a second, merged into one run directory (58/58 checks PASS on the merge; the pii_spans~vllm
point excluded from the generator because its unit is unmeasured by design); CPU suite at 10^4 (15/15 PASS); faults job with phases measure and oom
(11/11 checks PASS); the drift verdict of the earlier faults job at c9e8b025 recomputed offline (PASS). Tables below are produced by
the T20R-10 analysis script from the retrieved result files; "stratum" is the probe's length stratum 0..2 (shortest first), per-job seconds are
`wall_s / jobs` of warm `ok` probe requests (`since_load >= warmup_requests`, the profile's value), spread is half the interquartile range over the median
(a stratum under four requests has none).

## Headline per unit (1000 items; signal.resample 10000)

| unit | per-item s (throughput wall) | worker s per job | load_s | probe warm s of 900 | single r, u | single spread | B1 per-job at middle stratum | B2 per-job | B* | device resident, m0, m1 |
|---|---|---|---|---|---|---|---|---|---|---|
| speech_to_text~transformers | 0.368 | 0.340 | 14.6 | 40.4 | 0.113, 0.109 | 0.80 | 0.614 | 0.229 (B=19) | 19 | 1.62 GB, 44 MB, 6.4 MB per padded s |
| speech_to_text~crisperwhisper | 1.831 | 1.740 | 0.05 | 685 | 1.985, 0.0160 | 0.011 | 2.058 | 2.034 (B=6) | 6 | 1.65 GB, 3.0 GB, 0 |
| speaker_embeddings~speechbrain | 0.0531 | 0.0131 | 8.9 | 12.9 | 0.0199, 9.2e-5 | 0.085 | 0.0204 | 0.0108 (B=9) | 9 / 90 | 123 MB, 67 MB, 0.139 MB per padded s |
| diarize~pyannote | 0.103 | 0.0376 | 26.5 | 32.4 | 0.0242, 1.1e-4 | 0.013 | 0.0247 | 0.0227 (B=9) | 90 / 9 | 67 MB, 159 MB, 0 |
| pii_spans~transformers | 10.0 | 8.36 | 23.5 | 500.6 | 4.23, 6.1e-4 | 0.98 | 4.87 | 6.44 (B=2) | 1 | 64.5 GB, 0.85 GB, 28 KB per padded char |
| pii_spans~vllm | 3.513 | - | - | - | unmeasured | - | - | - | - | - |
| signal.resample~numpy-sinc (CPU) | 0.0410 | 0.0264 | 1e-5 | 0.58 | 0.00059, 0.0134 | 0.068 | 0.0276 | - | 1 | - |

`a` (the per-item cost beyond requests) is 0.104-0.261 s for every measured unit, CPU included (a n 12-30 items each).
B* is the chosen batch per stratum, listed in the per-unit table below.

## Per unit

| unit | point | status | r | u | a | a n | load_s | load n | B* | resident | m0 | m1 | probe warm s | probe_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diarize~pyannote | 1000 | measured | 0.02416 | 0.0001112 | 0.264 | 30 | 26.54 | 1 | 1.7-4.55:90,4.58-5.12:90,5.12-33.9:9 | 67112960 | 1.591e+08 | 0 | 32.42 | 900 |
| pii_spans~transformers | 1000 | measured | 4.23 | 0.0006101 | 0.2116 | 26 | 23.5 | 1 | 94-1.04e+03:1,1.04e+03-1.34e+03:1,1.36e+03-2.56e+03:1 | 64516900864 | 8.473e+08 | 2.803e+04 | 500.6 | 900 |
| pii_spans~vllm | 1000 | unmeasured | - | - | - | 0 | - | 0 | - | - | - | - | - | 900 |
| speaker_embeddings~speechbrain | 1000 | measured | 0.01993 | 9.247e-05 | 0.2376 | 30 | 8.876 | 1 | 1.7-4.55:9,4.58-5.12:9,5.12-33.9:90 | 123314688 | 6.714e+07 | 1.39e+05 | 12.93 | 900 |
| speech_to_text~crisperwhisper | 1000 | measured | 1.985 | 0.01598 | 0.2611 | 30 | 0.05022 | 1 | 1.7-4.55:6,4.58-5.39:6,5.42-31.6:6 | 1653349888 | 3e+09 | 0 | 685 | 900 |
| speech_to_text~transformers | 1000 | measured | 0.1129 | 0.1091 | 0.1951 | 30 | 14.56 | 1 | 1.7-4.55:19,4.58-5.54:19,5.57-16:19 | 1618667520 | 4.434e+07 | 6.416e+06 | 40.36 | 900 |
| signal.resample~numpy-sinc | 10000 | measured | 0.0005854 | 0.01343 | 0.1036 | 12 | 1.243e-05 | 1 | 1-2.01:1,2.01-2.01:1,2.01-3:1 | - | - | - | 0.5846 | 900 |

## Phase and stratum

| unit | phase | stratum | batch | n | per-job s | spread | units/job | padding | device peak B |
|---|---|---|---|---|---|---|---|---|---|
| diarize~pyannote | BATCHED | 0 | 9 | 3 | 0.02231 | - | 2.217 | 1.098 | 226241536 |
| diarize~pyannote | BATCHED | 0 | 90 | 3 | 0.02428 | - | 3.284 | 1.372 | 226241536 |
| diarize~pyannote | BATCHED | 1 | 9 | 3 | 0.02893 | - | 4.901 | 1.044 | 226241536 |
| diarize~pyannote | BATCHED | 1 | 90 | 3 | 0.02298 | - | 6.359 | 1.254 | 226241536 |
| diarize~pyannote | BATCHED | 2 | 9 | 3 | 0.02273 | - | 8.653 | 1.062 | 226241536 |
| diarize~pyannote | BATCHED | 2 | 90 | 3 | 0.06171 | - | 13.68 | 2.314 | 9051022336 |
| diarize~pyannote | SINGLE | 0 | 1 | 3 | 0.02433 | - | 1.74 | 1 | 226241536 |
| diarize~pyannote | SINGLE | 1 | 1 | 15 | 0.02468 | 0.009546 | 4.59 | 1 | 226241536 |
| diarize~pyannote | SINGLE | 2 | 1 | 3 | 0.02533 | - | 8.005 | 1 | 226241536 |
| pii_spans~transformers | BATCHED | 0 | 2 | 1 | 10.13 | - | 585.5 | 1.196 | 65444789248 |
| pii_spans~transformers | BATCHED | 0 | 4 | 3 | 7.086 | - | 661.5 | 1.475 | 65547513856 |
| pii_spans~transformers | BATCHED | 1 | 2 | 3 | 6.449 | - | 1271 | 1.056 | 65496083456 |
| pii_spans~transformers | BATCHED | 1 | 4 | 3 | 8.328 | - | 1325 | 1.126 | 65532816896 |
| pii_spans~transformers | BATCHED | 2 | 2 | 3 | 6.424 | - | 1808 | 1.077 | 65473420800 |
| pii_spans~transformers | BATCHED | 2 | 4 | 2 | 5.433 | - | 1975 | 1.222 | 6.552e+10 |
| pii_spans~transformers | SINGLE | 0 | 1 | 3 | 3.89 | - | 97 | 1 | 65016973312 |
| pii_spans~transformers | SINGLE | 1 | 1 | 14 | 6.85 | 0.6278 | 1050 | 1 | 6.549e+10 |
| pii_spans~transformers | SINGLE | 2 | 1 | 3 | 0.5801 | - | 1528 | 1 | 65395152384 |
| speaker_embeddings~speechbrain | BATCHED | 0 | 9 | 3 | 0.008853 | - | 2.217 | 1.098 | 171846656 |
| speaker_embeddings~speechbrain | BATCHED | 0 | 90 | 3 | 0.01105 | - | 3.284 | 1.372 | 204771328 |
| speaker_embeddings~speechbrain | BATCHED | 1 | 9 | 3 | 0.01211 | - | 4.901 | 1.044 | 213699584 |
| speaker_embeddings~speechbrain | BATCHED | 1 | 90 | 3 | 0.01265 | - | 6.359 | 1.254 | 249518592 |
| speaker_embeddings~speechbrain | BATCHED | 2 | 9 | 3 | 0.01688 | - | 8.653 | 1.062 | 268895232 |
| speaker_embeddings~speechbrain | BATCHED | 2 | 90 | 3 | 0.01636 | - | 13.68 | 2.314 | 566606336 |
| speaker_embeddings~speechbrain | SINGLE | 0 | 1 | 3 | 0.02063 | - | 1.74 | 1 | 159693312 |
| speaker_embeddings~speechbrain | SINGLE | 1 | 1 | 15 | 0.01982 | 0.3355 | 4.59 | 1 | 204843008 |
| speaker_embeddings~speechbrain | SINGLE | 2 | 1 | 3 | 0.02235 | - | 8.005 | 1 | 234001408 |
| speech_to_text~crisperwhisper | BATCHED | 0 | 6 | 3 | 2.004 | - | 2.402 | 1.128 | 4653800960 |
| speech_to_text~crisperwhisper | BATCHED | 0 | 35 | 3 | 1.447 | - | 3.323 | 1.35 | 4653800960 |
| speech_to_text~crisperwhisper | BATCHED | 1 | 6 | 3 | 2.05 | - | 5.066 | 1.063 | 4653800960 |
| speech_to_text~crisperwhisper | BATCHED | 1 | 35 | 3 | 1.735 | - | 6.426 | 1.236 | 4653800960 |
| speech_to_text~crisperwhisper | BATCHED | 2 | 6 | 3 | 2.08 | - | 9.058 | 1.091 | 4653800960 |
| speech_to_text~crisperwhisper | BATCHED | 2 | 35 | 3 | 1.903 | - | 13.9 | 2.034 | 4653800960 |
| speech_to_text~crisperwhisper | SINGLE | 0 | 1 | 3 | 2.043 | - | 1.74 | 1 | 4653800960 |
| speech_to_text~crisperwhisper | SINGLE | 1 | 1 | 15 | 2.049 | 0.00666 | 4.59 | 1 | 4653800960 |
| speech_to_text~crisperwhisper | SINGLE | 2 | 1 | 3 | 2.127 | - | 8.005 | 1 | 4653800960 |
| speech_to_text~transformers | BATCHED | 0 | 4 | 3 | 0.1652 | - | 2.486 | 1.104 | 1883040256 |
| speech_to_text~transformers | BATCHED | 0 | 19 | 3 | 0.07722 | - | 3.173 | 1.248 | 2917174784 |
| speech_to_text~transformers | BATCHED | 1 | 4 | 3 | 0.2648 | - | 5.174 | 1.071 | 1894966784 |
| speech_to_text~transformers | BATCHED | 1 | 19 | 3 | 0.0967 | - | 6.449 | 1.228 | 2968199680 |
| speech_to_text~transformers | BATCHED | 2 | 4 | 3 | 0.4089 | - | 9.011 | 1.074 | 1909874688 |
| speech_to_text~transformers | BATCHED | 2 | 19 | 3 | 0.1745 | - | 11.7 | 1.346 | 3121361408 |
| speech_to_text~transformers | SINGLE | 0 | 1 | 3 | 0.3676 | - | 1.74 | 1 | 1689903616 |
| speech_to_text~transformers | SINGLE | 1 | 1 | 15 | 0.0945 | 3.742 | 4.59 | 1 | 1692187136 |
| speech_to_text~transformers | SINGLE | 2 | 1 | 3 | 1.262 | - | 8.005 | 1 | 1695384064 |
| signal.resample~numpy-sinc | SINGLE | 0 | 1 | 3 | 0.01276 | - | 1.001 | 1 | - |
| signal.resample~numpy-sinc | SINGLE | 1 | 1 | 15 | 0.02758 | 0.04516 | 2.01 | 1 | - |
| signal.resample~numpy-sinc | SINGLE | 2 | 1 | 3 | 0.04107 | - | 3.001 | 1 | - |

## B1, B2 and B*

| unit | stratum | length | B1 | B2 | B* |
|---|---|---|---|---|---|
| diarize~pyannote | 0 | 1.74 | 0.02436 | 0.02219 | 90 |
| diarize~pyannote | 1 | 4.59 | 0.02467 | 0.02269 | 90 |
| diarize~pyannote | 2 | 8.005 | 0.02505 | 0.02328 | 9 |
| pii_spans~transformers | 0 | 97 | 4.289 | 6.436 | 1 |
| pii_spans~transformers | 1 | 1050 | 4.87 | 6.436 | 1 |
| pii_spans~transformers | 2 | 1528 | 5.162 | 6.436 | 1 |
| speaker_embeddings~speechbrain | 0 | 1.74 | 0.02009 | 0.008213 | 9 |
| speaker_embeddings~speechbrain | 1 | 4.59 | 0.02035 | 0.01082 | 9 |
| speaker_embeddings~speechbrain | 2 | 8.005 | 0.02067 | 0.01394 | 90 |
| speech_to_text~crisperwhisper | 0 | 1.74 | 2.013 | 2.009 | 6 |
| speech_to_text~crisperwhisper | 1 | 4.59 | 2.058 | 2.034 | 6 |
| speech_to_text~crisperwhisper | 2 | 8.005 | 2.113 | 2.064 | 6 |
| speech_to_text~transformers | 0 | 1.74 | 0.3029 | 0.1292 | 19 |
| speech_to_text~transformers | 1 | 4.59 | 0.6139 | 0.2289 | 19 |
| speech_to_text~transformers | 2 | 8.005 | 0.9867 | 0.3483 | 19 |
| signal.resample~numpy-sinc | 0 | 1.001 | 0.01403 | - | 1 |
| signal.resample~numpy-sinc | 1 | 2.01 | 0.02759 | - | 1 |
| signal.resample~numpy-sinc | 2 | 3.001 | 0.0409 | - | 1 |

## Post-load requests 0..10

Batch-1 requests by `since_load`, wall over the single fit's prediction at that length (the load request, since_load 0, includes the load).
One worker per probe, so each row is one request, not a median; the table is the pooled probe requests, one per `since_load`.

| unit | since_load | median ratio | n |
|---|---|---|---|
| diarize~pyannote | 0 | 30.29 | 1 |
| diarize~pyannote | 1 | 1.194 | 1 |
| diarize~pyannote | 2 | 1.193 | 1 |
| diarize~pyannote | 3 | 0.9988 | 1 |
| diarize~pyannote | 4 | 0.995 | 1 |
| diarize~pyannote | 5 | 1.011 | 1 |
| diarize~pyannote | 6 | 1.018 | 1 |
| diarize~pyannote | 7 | 0.9957 | 1 |
| diarize~pyannote | 8 | 0.9998 | 1 |
| diarize~pyannote | 9 | 0.9893 | 1 |
| diarize~pyannote | 10 | 0.9982 | 1 |
| pii_spans~transformers | 0 | 0.9597 | 1 |
| pii_spans~transformers | 1 | 0.122 | 1 |
| pii_spans~transformers | 2 | 0.1195 | 1 |
| pii_spans~transformers | 3 | 0.907 | 1 |
| pii_spans~transformers | 4 | 1.343 | 1 |
| pii_spans~transformers | 5 | 0.5168 | 1 |
| pii_spans~transformers | 6 | 0.1225 | 1 |
| pii_spans~transformers | 7 | 2.053 | 1 |
| pii_spans~transformers | 8 | 0.5179 | 1 |
| pii_spans~transformers | 9 | 1.733 | 1 |
| pii_spans~transformers | 10 | 1.246 | 1 |
| speaker_embeddings~speechbrain | 0 | 30.31 | 1 |
| speaker_embeddings~speechbrain | 1 | 1.71 | 1 |
| speaker_embeddings~speechbrain | 2 | 1.047 | 1 |
| speaker_embeddings~speechbrain | 3 | 1.023 | 1 |
| speaker_embeddings~speechbrain | 4 | 1.026 | 1 |
| speaker_embeddings~speechbrain | 5 | 1.059 | 1 |
| speaker_embeddings~speechbrain | 6 | 0.3907 | 1 |
| speaker_embeddings~speechbrain | 7 | 1.057 | 1 |
| speaker_embeddings~speechbrain | 8 | 1.558 | 1 |
| speaker_embeddings~speechbrain | 9 | 0.9928 | 1 |
| speaker_embeddings~speechbrain | 10 | 0.3628 | 1 |
| speech_to_text~crisperwhisper | 0 | 7.8 | 1 |
| speech_to_text~crisperwhisper | 1 | 0.9992 | 1 |
| speech_to_text~crisperwhisper | 2 | 1.002 | 1 |
| speech_to_text~crisperwhisper | 3 | 1.015 | 1 |
| speech_to_text~crisperwhisper | 4 | 1.015 | 1 |
| speech_to_text~crisperwhisper | 5 | 1.013 | 1 |
| speech_to_text~crisperwhisper | 6 | 1.013 | 1 |
| speech_to_text~crisperwhisper | 7 | 0.9941 | 1 |
| speech_to_text~crisperwhisper | 8 | 1.014 | 1 |
| speech_to_text~crisperwhisper | 9 | 0.9843 | 1 |
| speech_to_text~crisperwhisper | 10 | 0.9788 | 1 |
| speech_to_text~transformers | 0 | 7.819 | 1 |
| speech_to_text~transformers | 1 | 1.1 | 1 |
| speech_to_text~transformers | 2 | 1.347 | 1 |
| speech_to_text~transformers | 3 | 1 | 1 |
| speech_to_text~transformers | 4 | 1.186 | 1 |
| speech_to_text~transformers | 5 | 1.418 | 1 |
| speech_to_text~transformers | 6 | 1.301 | 1 |
| speech_to_text~transformers | 7 | 1 | 1 |
| speech_to_text~transformers | 8 | 1.144 | 1 |
| speech_to_text~transformers | 9 | 0.1358 | 1 |
| speech_to_text~transformers | 10 | 0.1317 | 1 |
| signal.resample~numpy-sinc | 0 | 1.148 | 1 |
| signal.resample~numpy-sinc | 1 | 1.024 | 1 |
| signal.resample~numpy-sinc | 2 | 0.9096 | 1 |
| signal.resample~numpy-sinc | 3 | 0.8476 | 1 |
| signal.resample~numpy-sinc | 4 | 1.183 | 1 |
| signal.resample~numpy-sinc | 5 | 1.241 | 1 |
| signal.resample~numpy-sinc | 6 | 1.046 | 1 |
| signal.resample~numpy-sinc | 7 | 1.034 | 1 |
| signal.resample~numpy-sinc | 8 | 0.9225 | 1 |
| signal.resample~numpy-sinc | 9 | 1 | 1 |
| signal.resample~numpy-sinc | 10 | 1.036 | 1 |

Least w by the rule (median of requests w+1..w+3 inside the steady band of the later requests), per unit: pyannote 1; the other four GPU units 0.
The three noisy units (stt transformers, speechbrain, pii transformers) have single requests at 0.12-2.0 of the fit after the load, so the rule is
inconclusive for them (their steady band is wider than the decay).

## Unit finding: speech_to_text~transformers walls are bimodal and the single fit can have a negative slope

At 3262a713 the faults job's measure phase (100 utterances) fitted batch 1 as r = 0.577 s, u = None (the Theil-Sen slope was negative, so the fit fell
back to the median wall), spread 1.169, n 18. The warm batch-1 walls at nearly the same audio length differ tenfold: audio 5.6-6.4 s gives 0.096, 0.098, 0.101,
0.102, 0.112, 0.193, 0.251 s and also 0.787, 0.948, 1.021, 1.056, 1.248, 1.256 s. A production window under two workers in the oom phase gave rho 0.33,
3.52, 0.95, 5.11, 2.10, 3.89 (median 2.81 against a band of -0.54..2.54), so a k = 1 drift record was written there. In the GPU table point the fit is r 0.113, u 0.109
with spread 0.80, and the per-job medians by stratum are 0.368, 0.0945 and 1.262 s at 1.7, 4.6 and 8.0 s of audio: the line passes through none
of them (0.30, 0.61, 0.99). The cause of the two modes is not diagnosed (generation length to the token limit against an early stop is a candidate;
contention between the pool's workers is not it, since the measure phase has one). The drift test's band is built from this spread (half-width =
`drift_confidence` x sqrt(pi/2) x spread / sqrt(window)): ±156 % at spread 1.18, so it can only flag a slowdown above about 2.6x. pii_spans~transformers is the same
(spread 0.98, strata medians 3.9, 6.9 and 0.58 s). crisperwhisper (0.011), pyannote (0.013) and speechbrain (0.085) are tight, and the CPU unit is 0.068.

## Batching (R-237)

At 10^3 items the held jobs fit the memory batch, and B1 and B2 were measured per stratum:
- stt transformers: B = 19, 2.3x faster per job than B = 1 at the middle stratum (0.229 against 0.614 s); device peak grows 1.69 -> 2.92-3.12 GB (m1 6.4 MB per padded audio second).
- speechbrain: B = 9 is 2.3x faster per job at stratum 0 (0.0089 against 0.0206 s) and 1.6x at the middle (0.0121 against 0.0198); B = 90 is about as fast as 9 at the first strata (0.0111, 0.0127), 2.2x faster at the longest (0.0164 against 0.0224 at B = 1).
- pyannote: at most 10 % faster than B = 1 (B = 9 per job 0.0223, 0.0289, 0.0227 against 0.0243, 0.0247, 0.0253 s; B = 90 0.0243, 0.0230, 0.0617), slower at 90 in the longest stratum (padding 2.3, peak 9.05 GB).
- crisperwhisper: per-job 2.0-2.1 s at B = 1 and 6, 1.45-1.90 s at B = 35; device peak is 4.65 GB at every batch (m1 = 0). The batch is not visible in memory.
- pii_spans~transformers: B = 2 and 4 take 5.4-10.1 s per job against a B = 1 fit of 4.3-5.2 s, so no gain; B = 1 chosen; peak 65 GB at every batch.
Padding by stratum (padded over real units): 1.04-1.13 at B = 4-9, 1.25-1.37 at B = 19-90 for short strata, 2.0-2.3 for the longest stratum at B >= 35.

## Warm work against probe_s

The probe's warm requests (since_load above the warmup, ok) sum to 0.6-685 s of the 900 s budget (`point_wall_s`): crisperwhisper 685 s and pii transformers 501 s are
near it; stt 40 s, pyannote 32 s, speechbrain 13 s, CPU 0.6 s. A probe at 923 items is not what bounds these units; the seconds budget is.

## R-229: crisperwhisper at batch 1 beside e801bc6c

Batch-1 warm requests (n 21): 2.043, 2.049 and 2.127 s per job at strata 0, 1, 2 (audio 1.7, 4.6 and 8.0 s), spread 0.011; `job_s` 2.054 s over the ladder. e801bc6c
read 1.143 s and 868c06df 2.439 s. Engine `transformers`, `confidence true`, `max_new_tokens 256` (recorded). Wall does not follow audio length (1.7 s of audio costs 2.04 s), and device peak
is 4.65 GB with a resident 1.65 GB. The steady figure is now 2.05 s, so the 868c06df reading was not an outlier of that run; why e801bc6c read 1.14 s is not
established (the env locks and load dtype of the two builds were not compared).

## Local-weights digest cost (carry c)

Cold bind of a local snapshot (every file hashed) against warm (state stat'ed, identity read), per snapshot: gemma-4-31B w4a16 (23.3 GB, 9 files) 16.2 s cold, 1.5 ms warm;
CrisperWhisper turbo (1.62 GB, 14 files) 1.08 s; whisper-large-v3-turbo (1.62 GB, 11 files) 1.02 s; ecapa (89 MB, 10 files) 0.06 s. About 0.7 s per GB cold (page cache warm after the model load), then constant.

## resolve_s (carry c)

Not measured: the faults verdict that carried the worker start-to-ready-mark seconds failed to write (a harness bug fixed afterwards), and the ready marks keep
the process start in boot ticks without the boot time. The
key stays a bound (a worker's start includes env builds, which depend on the cold state of a group's env root), not a fitted figure.

## OOM and bisect on the request path (carry a), job with phases measure and oom

The hook wraps `senselab_worker.serve.call_handle` in every worker env (`.pth`), returns an `OutOfMemoryError` error for each item of a request of more than 5 items, and writes a `fired` mark first.
Result: 6 marks, all at request size 8; one `oom` record at batch 8, length 3.775, co-tenancy 1, grant 85.9 GB, with its worker and the basis probe; the seed record
(batch 2^20) is the trigger; requests after the fault were answered warm at batches up to the limit; no job was lost (the pool split the request in halves until a half fit). Eleven of
eleven checks PASS. The injection fired on the real request path, so the pool's handling is proven; the backend's own memory is not what faulted.

## Drift under two workers (carry b), earlier faults job at c9e8b025, recomputed offline

39 batch-1 readings at co-tenancy 2, window 6, band half-width 1.5595, median rho 1.149; replaying `telemetry.Drift` over the readings in commit order gives no drifted window and the run wrote no
drift record, so "no record" lies inside the band. The band is wide because of the unit finding above.

## Single-item rule data

| unit | a | per-job s at middle stratum | k |
|---|---|---|---|
| diarize~pyannote | 0.264 | 0.02467 | 42 |
| pii_spans~transformers | 0.2116 | 4.87 | 2 |
| pii_spans~vllm | - | - | - |
| speaker_embeddings~speechbrain | 0.2376 | 0.02035 | 46 |
| speech_to_text~crisperwhisper | 0.2611 | 2.058 | 2 |
| speech_to_text~transformers | 0.1951 | 0.6139 | 2 |
| signal.resample~numpy-sinc | 0.1036 | 0.02759 | 15 |

`a` is 0.10-0.26 s on every unit; k = floor(4a/t) at the middle stratum runs from 2 (units with t above about 0.4 s) to 46 (speechbrain, t 0.02 s).

## Proposals for the three keys (a proposal, not a code change; none is fitted from this campaign)

All code references are at 6f30fec3.

**warmup_requests** (now 2, `probe.py:257` and `:771`, `telemetry.py:43`).
- The probe can measure it per resource, but not with today's pure plan: `Probe.plan` puts `warmup_requests` WARMUP items first (`probe.py:771-780`) and the estimator drops requests under it
  (`probe.py:257`). Every `Observation` carries `since_load`, so the decay is observable; an adaptive form needs a WARMUP item that continues until the last three requests of a state agree within the unit's own band,
  which makes the plan sequential.
- Evidence for a default: the load request carries all of the cold cost (7.8-30x the steady fit, and it is excluded by `load_s`); the request after it is 0.99-1.0 (crisperwhisper), 1.19 (pyannote), 1.1-1.7
  (stt, speechbrain, noisy); the rule's least w is 0 for four GPU units and 1 for pyannote. The cluster-independent statement is "the first request after a load is cold, later ones are warm within noise".
- Proposed default and rule: `warmup_requests` 1 as the stated floor (the request after the load is excluded), with a diagnostic flag on an Estimate whose requests 1..3 differ from the later median beyond the
  fit's band; a group that sees the flag raises the key in its own config. The current 2 is a safe choice at a cost of one extra request per probe; the evidence does not distinguish 1 from 2 for the noisy units.

**single_item_jobs** (now 4, `probe.py:492`; `a` is `Estimate.item_s`, `probe.py:262-268`).
- `a` is already measured per unit at run time: for each whole probe item it is the item's wall minus the sum of its requests' walls, over every admitted SINGLE item (n 12-30 here), not from a difference of
  1-job and k-job items. The key only shapes the one extra item per probe (`probe.py:490-493`).
- Evidence: `a` is 0.10-0.26 s for every unit, CPU and GPU, so it is a platform per-claim cost that the probe measures, not a property of the model. The k of the old rule (a at least a fifth of a k-job item) runs from
  2 for units whose per-job time exceeds about 0.4 s to 46 for the fastest; a single count cannot satisfy it for both ends.
- Proposed default and rule: 2 (the schema minimum, `execution_settings.py:175`): the least extra item that still differs from the 1-job items, cheapest for the slow units where a larger k costs minutes, and
  harmless for the fast ones because `a` is read per item. Whether `a` varies with the job count within an item is not tested by these data and is the thing to check before relying on it.

**probe_items** (now 200 on orcd, a bound: `probe.py:496`, `calibration.py:488`).
- The batch itself is already measured per resource at run time: the batched phase computes each stratum's cap from the unit's own device fit and the grant (`memory_batch(device, budget, hi, grant.quantum_items)`,
  `probe.py` ~825-835) and plans against `left = probe_items - used_jobs`; the seconds budget (`probe_s`) also limits it (`over_seconds`). So probe_items is only a ceiling on jobs.
- Evidence: measured memory batches differ by unit from 1 to 90 (pii 1, crisperwhisper 6, speechbrain 9-90, stt 19, pyannote 9-90), the warm work used 0.6-685 s of 900 s, and the 923-item table profile was not the binding limit for any unit.
  A single target B therefore cannot be fitted from one resource.
- Proposed rule, derived from other keys and no measured B: `probe_items = warmup_requests + length_strata x min_requests x (1 + pool.quantum_items) + min_requests x single_item_jobs`, the jobs of the minimal protocol at the
  largest batch the pool allows. With the proposed keys (warmup 1, strata 3, min_requests 3, single_item_jobs 2) and a pool quantum of 100 that is 1 + 3 x 3 x 101 + 3 x 2 = 916. (With the orcd profile's `length_strata` 2 it is 1 + 2 x 3 x 101 + 6 = 613.)
