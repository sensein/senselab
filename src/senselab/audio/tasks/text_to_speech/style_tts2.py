"""Functionality for StyleTTS2.0 that can be used for text to speech using reference audios to mimic voice cloning.

For more details, see https://github.com/yl4579/StyleTTS2
"""

# Likely not needed but keeping for now
# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# load packages
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import librosa
import numpy as np
import phonemizer
import torch
import torchaudio
import yaml
from nltk.tokenize import word_tokenize

from senselab.audio.data_structures.audio import Audio

# load phonemizer
from senselab.libs.StyleTTS2.models import build_model, load_ASR_models, load_F0_models
from senselab.libs.StyleTTS2.Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from senselab.libs.StyleTTS2.text_utils import TextCleaner
from senselab.libs.StyleTTS2.utils import recursive_munch
from senselab.libs.StyleTTS2.Utils.PLBERT.util import load_plbert
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import StyleTTSModel


class StyleTTS2:
    """StyleTTS2 Model that can run inference as described in the StyleTTS2 documentation."""

    _pipelines: Dict[str, Any] = {}

    @classmethod
    def _get_styletts2_pipeline(
        cls,
        model: Literal["LibriTTS", "LJSpeech"] = "LibriTTS",
        # prematched_vocoder: bool,
        # topk: int,
        device: Optional[DeviceType] = None,
    ) -> "StyleTTS2":  # noqa: ANN401
        """Get or create a StyleTTS2.0 pipeline."""
        key = f"{model}-{device}"
        if key not in cls._pipelines:
            device, _ = _select_device_and_dtype(
                user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
            )
            style_tts = StyleTTS2(device, model)
            cls._pipelines[key] = style_tts
        return cls._pipelines[key]

    @classmethod
    def synthesize_texts(
        cls,
        texts: List[str],
        reference_audios: Union[List[Audio], Audio],
        model: StyleTTSModel = StyleTTSModel(path_or_uri="LibriTTS"),
        language: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_step: int = 5,
        embedding_scale: int = 1,
    ) -> List[Audio]:
        """Synthesize audios from texts with reference audios.

        TODO: Finish docstring
        """
        if language and language.alpha_2 != "en":
            raise ValueError(
                "Though StyleTTS2 provides some support for multiple languages, \
                    we currently only adopt the English version of the model."
            )
        if isinstance(reference_audios, List) and len(reference_audios) != len(texts):
            raise ValueError("List of reference audios must be the same size as the list of texts.")

        tts_model = StyleTTS2._get_styletts2_pipeline(model=model.path_or_uri, device=device)
        target_audios = reference_audios if isinstance(reference_audios, List) else [reference_audios] * len(texts)

        synthesized_speech = []

        for i in range(len(texts)):
            audio = target_audios[i]
            text = texts[i]
            ref_style = tts_model._compute_style(audio)
            wav = tts_model._inference(
                text, ref_style, alpha=alpha, beta=beta, diffusion_steps=diffusion_step, embedding_scale=embedding_scale
            )
            synthesized_speech.append(Audio(waveform=wav, sampling_rate=24000))
        return synthesized_speech

    def __init__(self, device: DeviceType, model: Literal["LibriTTS", "LJSpeech"] = "LibriTTS") -> None:
        """Initializes a StyleTTS2 model.

        TODO write remainder of this
        """
        self.textclenaer = TextCleaner()

        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4  # some magic numbers
        self.device = device

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        config = yaml.safe_load(open("libs/StyleTTS2/Models/LibriTTS/config.yml"))
        # load pretrained ASR model
        ASR_config = config.get("ASR_config", False)
        ASR_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model

        BERT_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path)

        self.model_params = recursive_munch(config["model_params"])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(device) for key in self.model]

        params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location="cpu")
        params = params_whole["net"]

        for key in self.model:
            if key in params:
                print("%s loaded" % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:  # noqa: E722
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False,
        )

    def _length_to_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """Generates a mask of a specific length."""
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def _preprocess(self, wave: np.ndarray) -> torch.Tensor:
        """Preprocesses the audio."""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def _compute_style(self, ref: Audio) -> torch.Tensor:
        """Computes the style of the audio which can then be applied to any voice/speech."""
        wave, sr = ref.waveform, ref.sampling_rate
        audio, index = librosa.effects.trim(wave.numpy(), top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self._preprocess(audio).to(self.device.value)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def _inference(
        self,
        text: str,
        ref_s: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
    ) -> np.ndarray:
        """Runs generic inference of StyleTTS2."""
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device.value).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device.value)
            text_mask = self._length_to_mask(input_lengths).to(self.device.value)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device.value),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later

    def _LFinference(
        self,
        text: str,
        s_prev: torch.Tensor,
        ref_s: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        t: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Runs Long Form Inference of StyleTTS2 for large text inputs."""
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        ps = ps.replace("``", '"')
        ps = ps.replace("''", '"')

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device.value).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device.value)
            text_mask = self._length_to_mask(input_lengths).to(self.device.value)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device.value),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps,
            ).squeeze(1)

            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[
            ..., :-100
        ], s_pred  # weird pulse at the end of the model, need to be fixed later

    def _STinference(
        self,
        text: str,
        ref_s: torch.Tensor,
        ref_text: str,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: int = 1,
    ) -> np.ndarray:
        """Runs style transfer of a given text that has strong emotional aspects."""
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device.value).unsqueeze(0)

        ref_text = ref_text.strip()
        ps = self.global_phonemizer.phonemize([ref_text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)

        ref_tokens = self.textclenaer(ps)
        ref_tokens.insert(0, 0)
        ref_tokens = torch.LongTensor(ref_tokens).to(self.device.value).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device.value)
            text_mask = self._length_to_mask(input_lengths).to(self.device.value)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(self.device.value)
            ref_text_mask = self._length_to_mask(ref_input_lengths).to(self.device.value)
            _ref_bert_dur = self.model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device.value),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # reference from the same speaker as the embedding
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device.value)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later
