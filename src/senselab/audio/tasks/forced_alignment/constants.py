"""Provides constants for align functionality."""

SAMPLE_RATE = 16000

MINIMUM_SEGMENT_SIZE = 400

PUNKT_ABBREVIATIONS = ["dr", "vs", "mr", "mrs", "prof"]

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": ("WAV2VEC2_ASR_BASE_960H", "main"),
    "fr": ("VOXPOPULI_ASR_BASE_10K_FR", "main"),
    "de": ("VOXPOPULI_ASR_BASE_10K_DE", "main"),
    "es": ("VOXPOPULI_ASR_BASE_10K_ES", "main"),
    "it": ("VOXPOPULI_ASR_BASE_10K_IT", "main"),
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "main"),
    "zh": ("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", "main"),
    "nl": ("jonatasgrosman/wav2vec2-large-xlsr-53-dutch", "main"),
    "uk": ("Yehor/wav2vec2-xls-r-300m-uk-with-small-lm", "main"),
    "pt": ("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", "main"),
    "ar": ("jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "main"),
    "cs": ("comodoro/wav2vec2-xls-r-300m-cs-250", "main"),
    "ru": ("jonatasgrosman/wav2vec2-large-xlsr-53-russian", "main"),
    "pl": ("jonatasgrosman/wav2vec2-large-xlsr-53-polish", "main"),
    "hu": ("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "main"),
    "fi": ("jonatasgrosman/wav2vec2-large-xlsr-53-finnish", "main"),
    "fa": ("jonatasgrosman/wav2vec2-large-xlsr-53-persian", "main"),
    "el": ("jonatasgrosman/wav2vec2-large-xlsr-53-greek", "main"),
    "tr": ("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "main"),
    "da": ("saattrupdan/wav2vec2-xls-r-300m-ftspeech", "main"),
    "he": ("imvladikon/wav2vec2-xls-r-300m-hebrew", "main"),
    "vi": ("nguyenvulebinh/wav2vec2-base-vi", "main"),
    "ko": ("kresnik/wav2vec2-large-xlsr-korean", "main"),
    "ur": ("kingabzpro/wav2vec2-large-xls-r-300m-Urdu", "main"),
    "te": ("anuragshas/wav2vec2-large-xlsr-53-telugu", "main"),
    "hi": ("theainerd/Wav2Vec2-large-xlsr-hindi", "main"),
    "ca": ("softcatala/wav2vec2-large-xlsr-catala", "main"),
    "ml": ("gvs/wav2vec2-large-xlsr-malayalam", "main"),
    "no": ("NbAiLab/nb-wav2vec2-1b-bokmaal", "main"),
    "nn": ("NbAiLab/nb-wav2vec2-300m-nynorsk", "main"),
}
