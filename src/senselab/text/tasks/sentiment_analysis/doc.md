Text sentiment and emotion classification.

`analyze_sentiment` scores texts as negative / neutral / positive and returns the full label
distribution plus a signed valence (`p_pos - p_neg`). `classify_text_emotions` returns the full
distribution over seven emotions. Both run in-process on CPU, CUDA or MPS, cache loaded models, and
truncate long texts or, with `long_text="window"`, average over overlapping token windows.
