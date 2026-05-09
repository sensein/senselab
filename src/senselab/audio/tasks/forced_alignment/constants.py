"""Provides constants for align functionality."""

SAMPLE_RATE = 16000

MINIMUM_SEGMENT_SIZE = 400

PUNKT_ABBREVIATIONS = ["dr", "vs", "mr", "mrs", "prof"]

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

# MMS (Massively Multilingual Speech) — single-model fallback aligner
# covering 1100+ languages via per-language adapters. Selected via the
# ``align_transcriptions(..., aligner_model=MMS_MODEL_ID)`` knob; not the
# default for the languages already in DEFAULT_ALIGN_MODELS_HF /
# DEFAULT_ALIGN_MODELS_TORCH (preserves backwards-compat for existing
# callers, who keep getting the same per-language wav2vec2 models).
#
# Weights are CC-BY-NC 4.0 (research / non-commercial). Override via
# the public API for commercial-friendly alternatives.
MMS_MODEL_ID = "facebook/mms-1b-all"

# ISO 639-1 -> ISO 639-3 map for MMS adapter selection. MMS adapters are
# keyed by ISO-3 codes; senselab callers (and Language objects) typically
# carry ISO-1. Languages absent from this map cannot be auto-resolved to
# an MMS adapter and will raise an actionable error.
ISO_1_TO_3 = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ja": "jpn",
    "zh": "cmn",  # Mandarin Chinese
    "nl": "nld",
    "uk": "ukr",
    "ar": "ara",
    "cs": "ces",
    "ru": "rus",
    "pl": "pol",
    "hu": "hun",
    "fi": "fin",
    "fa": "fas",
    "el": "ell",
    "tr": "tur",
    "da": "dan",
    "he": "heb",
    "vi": "vie",
    "ko": "kor",
    "ur": "urd",
    "te": "tel",
    "hi": "hin",
    "ca": "cat",
    "ml": "mal",
    "no": "nor",
    "nn": "nno",
}

DEFAULT_ALIGN_MODELS_TORCH = {
    "fr": {"path_or_uri": "VOXPOPULI_ASR_BASE_10K_FR", "revision": "main"},
    "de": {"path_or_uri": "VOXPOPULI_ASR_BASE_10K_DE", "revision": "main"},
    "es": {"path_or_uri": "VOXPOPULI_ASR_BASE_10K_ES", "revision": "main"},
    "it": {"path_or_uri": "VOXPOPULI_ASR_BASE_10K_IT", "revision": "main"},
}

DEFAULT_ALIGN_MODELS_HF = {
    "en": {"path_or_uri": "facebook/wav2vec2-base-960h", "revision": "main"},
    "ja": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "revision": "main"},
    "zh": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "revision": "main"},
    "nl": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "revision": "main"},
    "uk": {"path_or_uri": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm", "revision": "main"},
    "pt": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "revision": "main"},
    "ar": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "revision": "main"},
    "cs": {"path_or_uri": "comodoro/wav2vec2-xls-r-300m-cs-250", "revision": "main"},
    "ru": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-russian", "revision": "main"},
    "pl": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-polish", "revision": "main"},
    "hu": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "revision": "main"},
    "fi": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "revision": "main"},
    "fa": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-persian", "revision": "main"},
    "el": {"path_or_uri": "jonatasgrosman/wav2vec2-large-xlsr-53-greek", "revision": "main"},
    "tr": {"path_or_uri": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "revision": "main"},
    "da": {"path_or_uri": "saattrupdan/wav2vec2-xls-r-300m-ftspeech", "revision": "main"},
    "he": {"path_or_uri": "imvladikon/wav2vec2-xls-r-300m-hebrew", "revision": "main"},
    "vi": {"path_or_uri": "nguyenvulebinh/wav2vec2-base-vi", "revision": "main"},
    "ko": {"path_or_uri": "kresnik/wav2vec2-large-xlsr-korean", "revision": "main"},
    "ur": {"path_or_uri": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu", "revision": "main"},
    "te": {"path_or_uri": "anuragshas/wav2vec2-large-xlsr-53-telugu", "revision": "main"},
    "hi": {"path_or_uri": "theainerd/Wav2Vec2-large-xlsr-hindi", "revision": "main"},
    "ca": {"path_or_uri": "softcatala/wav2vec2-large-xlsr-catala", "revision": "main"},
    "ml": {"path_or_uri": "gvs/wav2vec2-large-xlsr-malayalam", "revision": "main"},
    "no": {"path_or_uri": "NbAiLab/nb-wav2vec2-1b-bokmaal", "revision": "main"},
    "nn": {"path_or_uri": "NbAiLab/nb-wav2vec2-300m-nynorsk", "revision": "main"},
}
