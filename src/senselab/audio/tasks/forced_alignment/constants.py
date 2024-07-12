"""Provides constants for align functionality."""

from senselab.utils.data_structures.model import HFModel, TorchAudioModel

SAMPLE_RATE = 16000

MINIMUM_SEGMENT_SIZE = 400

PUNKT_ABBREVIATIONS = ["dr", "vs", "mr", "mrs", "prof"]

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "fr": TorchAudioModel(path_or_uri="VOXPOPULI_ASR_BASE_10K_FR", revision="main"),
    "de": TorchAudioModel(path_or_uri="VOXPOPULI_ASR_BASE_10K_DE", revision="main"),
    "es": TorchAudioModel(path_or_uri="VOXPOPULI_ASR_BASE_10K_ES", revision="main"),
    "it": TorchAudioModel(path_or_uri="VOXPOPULI_ASR_BASE_10K_IT", revision="main"),
}

DEFAULT_ALIGN_MODELS_HF = {
    "en": HFModel(path_or_uri="facebook/wav2vec2-base-960h", revision="main"),
    "ja": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-japanese", revision="main"),
    "zh": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", revision="main"),
    "nl": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-dutch", revision="main"),
    "uk": HFModel(path_or_uri="Yehor/wav2vec2-xls-r-300m-uk-with-small-lm", revision="main"),
    "pt": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", revision="main"),
    "ar": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-arabic", revision="main"),
    "cs": HFModel(path_or_uri="comodoro/wav2vec2-xls-r-300m-cs-250", revision="main"),
    "ru": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-russian", revision="main"),
    "pl": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-polish", revision="main"),
    "hu": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", revision="main"),
    "fi": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-finnish", revision="main"),
    "fa": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-persian", revision="main"),
    "el": HFModel(path_or_uri="jonatasgrosman/wav2vec2-large-xlsr-53-greek", revision="main"),
    "tr": HFModel(path_or_uri="mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", revision="main"),
    "da": HFModel(path_or_uri="saattrupdan/wav2vec2-xls-r-300m-ftspeech", revision="main"),
    "he": HFModel(path_or_uri="imvladikon/wav2vec2-xls-r-300m-hebrew", revision="main"),
    "vi": HFModel(path_or_uri="nguyenvulebinh/wav2vec2-base-vi", revision="main"),
    "ko": HFModel(path_or_uri="kresnik/wav2vec2-large-xlsr-korean", revision="main"),
    "ur": HFModel(path_or_uri="kingabzpro/wav2vec2-large-xls-r-300m-Urdu", revision="main"),
    "te": HFModel(path_or_uri="anuragshas/wav2vec2-large-xlsr-53-telugu", revision="main"),
    "hi": HFModel(path_or_uri="theainerd/Wav2Vec2-large-xlsr-hindi", revision="main"),
    "ca": HFModel(path_or_uri="softcatala/wav2vec2-large-xlsr-catala", revision="main"),
    "ml": HFModel(path_or_uri="gvs/wav2vec2-large-xlsr-malayalam", revision="main"),
    "no": HFModel(path_or_uri="NbAiLab/nb-wav2vec2-1b-bokmaal", revision="main"),
    "nn": HFModel(path_or_uri="NbAiLab/nb-wav2vec2-300m-nynorsk", revision="main"),
}
