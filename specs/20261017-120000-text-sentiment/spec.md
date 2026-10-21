# Text sentiment and emotion classification

## Scope
`senselab.text.tasks.sentiment_analysis`: `analyze_sentiment` and `classify_text_emotions`, in-process
Hugging Face sequence classification returning the full label distribution per text.

## Model choice
| Task | Default | Labels (index order) | Reference |
|---|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | negative, neutral, positive | Loureiro et al. 2022, TimeLMs (arXiv:2202.03829); Barbieri et al. 2020, TweetEval (arXiv:2010.12421) |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise | Hartmann 2022, model card (Ekman six + neutral) |

Both are small (RoBERTa-base and DistilRoBERTa-base, ~0.5 GB together) and run on CPU, CUDA and MPS.
Valence is `p_positive - p_negative`, in [-1, 1]; it keeps the neutral mass visible instead of
collapsing it into a hard label. Custom models are supported if they expose exactly one `pos*` and
one `neg*` label (sentiment) or any label set (emotion).

## Behaviour
- Revision: loads resolve the commit SHA via `resolve_model` (or `HFModel.commit_sha`) and load with
  `revision=<sha>`; the result carries `model_id` and `commit_sha`.
- Cache: models keyed by (repo, commit, device); tokenizers by (repo, commit).
- Device: CUDA, then MPS, then CPU unless the caller names one.
- Long texts: `truncate` (default) keeps the first model-length tokens and sets `truncated`;
  `window` splits into overlapping windows (`window_stride` tokens of overlap) and averages the
  distributions weighted by window length (`n_windows` reports the count).
- Empty or whitespace-only text: result at the same position with empty `scores`, no label, `n_windows=0`.
- Batches are formed over windows sorted by length; scores are independent of `batch_size` to ~1e-4.

## Limitations
- Sentiment was trained on tweets; transcripts of spoken conversation differ (disfluencies, no
  punctuation or casing from ASR, long turns, backchannels). Treat scores as a weak lexical-affective
  signal, not a validated measure for clinical speech.
- The emotion model was trained on mixed English corpora (text, dialogue, TV scripts); English only.
- Text-only: prosody and voice carry affect that these models never see.
- Window averaging is a heuristic; it is not calibrated against whole-turn human ratings.

## Measurement
1000 short sentences, M1 Pro, MPS, batch 32, default sentiment model, warm cache: ~1.98 s (~506 sentences/s).
