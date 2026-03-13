[![Build](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml/badge.svg)](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml)
[![codecov](https://codecov.io/gh/sensein/senselab/graph/badge.svg?token=9S8WY128PO)](https://codecov.io/gh/sensein/senselab)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/senselab.svg)](https://pypi.org/project/senselab/)
[![Python Version](https://img.shields.io/pypi/pyversions/senselab)](https://pypi.org/project/senselab)
[![License](https://img.shields.io/pypi/l/senselab)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/senselab)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sensein/senselab)

# ```senselab```
This Python package **streamlines**, **optimizes**, and **enforces best open-science practices** for processing and analyzing _behavioral data_ (primarily voice and speech, but also text and video) using robust reproducible pipelines and utilities.

## Quick start
```Python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios

audio = Audio(filepath='path_to_audio_file.wav')
print(audio.sampling_rate)
# ➡️ 44100

[resampled_audio] = resample_audios([audio], resample_rate=16000)
print(resampled_audio.sampling_rate)
# ➡️ 16000

audio_features = extract_features_from_audios([audio])
print(audio_features[0].keys())
# ➡️ dict_keys(['opensmile', 'praat_parselmouth', 'torchaudio', 'torchaudio_squim', ...])

transcript = transcribe_audios([audio])
print(transcript)
# ➡️ "The quick brown fox jumps over the lazy dog."
```

For more detailed information, check out our [**Documentation**](https://sensein.group/senselab/senselab.html) and our [**Tutorials**](https://github.com/sensein/senselab/blob/main/tutorials/audio/00_getting_started.ipynb).

💡 **Tip**: Many tutorials include Google Colab badges and you can try them instantly without installing anything on your local machine.



### Why should you use ```senselab```?
- **Modular design**: Easily integrate or use standalone transformations for flexible data manipulation.
- **Pre-built pipelines**: Access pre-configured pipelines to reduce setup time and effort.
- **Reproducibility**: Ensure consistent and verifiable results with fixed seeds and version-controlled steps.
- **Easy integration**: Seamlessly fit into existing workflows with minimal configuration.
- **Extensible**: Modify and contribute custom transformations and pipelines to meet specific research needs.
- **Comprehensive documentation**: Detailed guides, examples, and documentation for all features and modules.
- **Performance optimized**: Efficiently process large datasets with optimized code and algorithms.
- **Interactive examples**: Jupyter notebooks provide practical examples for deriving insights from real-world datasets.
- **senselab AI**: Interact with your data through an AI-based chatbot. The AI agent generates and runs senselab-based code for you, making exploration easier and giving you both the results and the code used to produce them (perfect for quick experiments or for users who prefer not to code).


---

## ⚠️ System Requirements
1. **If on macOS, this package requires an ARM64 architecture** due to PyTorch 2.2.2+ dropping support for x86-64 on macOS.

    ❌ Unsupported systems include:
    - macOS (Intel x86-64)
    - Other platforms where dependencies are unavailable

    To check your system compatibility, please run this command:
    ```bash
    python -c "import platform; print(platform.machine())"
    ```

    If the output is:
    - `arm64` → ✅ Your system is compatible.
    - `x86_64` → ❌ Your system is not supported.

    If you attempt to install this package on an unsupported system, the installation or execution will fail.

2. `FFmpeg` is required by some audio and video dependencies (e.g., `torchaudio`). Please make sure you have `FFmpeg` properly installed on your machine before installing and using `senselab` (see [here](https://www.ffmpeg.org/download.html) for detailed platform-dependent instructions).

3. CUDA libraries matching the CUDA version expected by the PyTorch wheels (e.g., the latest pytorch 2.8 expects cuda-12.8). To install those with conda, please do:
  - ```conda config --add channels nvidia```
  - ```conda install -y nvidia/label/cuda-12.8.1::cuda-libraries-dev```
4. Docker is required and must be running for some video models (e.g., MediaPipe-based estimators).
Please follow the official installation instructions for your platform: [Install Docker](https://docs.docker.com/get-started/get-docker/).
5. Some functionalities rely on HuggingFace models, and increasingly, models require authentication and signed license agreements. Instructions on how to generate a Hugging Face access token can be found here: https://huggingface.co/docs/hub/security-tokens
  - You can provide your HuggingFace token either by exporting it in your shell:
    ```bash
    export HF_TOKEN=your_token_here
    ```
  - or by adding it to your `.env` file (see `.env.example` for reference).

---

## Installation
Install this package via:

```sh
pip install 'senselab[all]'
```

Or get the newest development version via:

```sh
pip install 'git+https://github.com/sensein/senselab.git#egg=senselab[all]'
```

If you want to install only audio dependencies, you do:
```sh
pip install 'senselab'
```

To install articulatory, video, text, and senselab-ai extras, please do:
```sh
pip install 'senselab[articulatory,video,text,senselab-ai]'
```

---

## senselab AI (our AI-based chatbot)

#### Development (with poetry)

```bash
poetry install --extras "senselab-ai"
poetry run senselab-ai
```

#### Production (with pip)

```bash
pip install 'senselab[senselab-ai]'
senselab-ai
```

Once started, you can open the provided JupyterLab interface, setup the agent and chat with it, and let it create and execute code for you.

![Example of how senselab-ai works](<tutorials/senselab-ai/resources/Screenshot 2025-09-02 at 8.52.31 PM.png>)

For a walkthrough, see: [`tutorials/senselab-ai/senselab_ai_intro.ipynb`](tutorials/senselab-ai/senselab_ai_intro.ipynb).


---

## Contributing
<ins>We welcome contributions from the community!</ins> Before proceeding with that, please review our [**CONTRIBUTING.md**](https://github.com/sensein/senselab/blob/main/CONTRIBUTING.md).

---

## Funding
`senselab` is mostly supported by the following organizations and initiatives:
- McGovern Institute ICON Fellowship
- NIH Bridge2AI Precision Public Health (OT2OD032720)
- Child Mind Institute
- ReadNet Project
- Chris and Lann Woehrle Psychiatric Fund

---

## Acknowledgments

`senselab` builds on the work of many open-source projects. We gratefully acknowledge the developers and maintainers of the following key dependencies:

* [PyTorch](https://github.com/pytorch/pytorch), [Torchvision](https://github.com/pytorch/vision), [Torchaudio](https://github.com/pytorch/audio)
_deep learning framework and audio/vision extensions_
* [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Accelerate](https://github.com/huggingface/accelerate), [Huggingface Hub](https://github.com/huggingface/huggingface_hub)
_training and inference utilities plus (pre-)trained models and datasets_
* [Scikit-learn](https://github.com/scikit-learn/scikit-learn), [UMAP-learn](https://github.com/lmcinnes/umap)
_machine learning utilities_
* [Matplotlib](https://github.com/matplotlib/matplotlib)
_visualization toolkit_
* [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth), [OpenSMILE](https://github.com/audeering/opensmile), [SpeechBrain](https://github.com/speechbrain/speechbrain), [SPARC](speech-articulatory-coding), [Pyannote-audio](https://github.com/pyannote/pyannote-audio), [Coqui-TTS](https://github.com/idiap/coqui-ai-TTS), [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [Vocos](https://github.com/gemelo-ai/vocos), [Audiomentations](https://github.com/iver56/audiomentations), [Torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
_speech and audio processing tools_
* [NLTK](https://github.com/nltk/nltk), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers), [Pylangacq](https://github.com/jacksonllee/pylangacq), [Jiwer](https://github.com/jitsi/jiwer)
_text and language processing tools_
* [OpenCV](https://github.com/opencv/opencv-python), [Ultralytics](https://github.com/ultralytics/ultralytics), [mediapipe](https://github.com/google-ai-edge/mediapipe), [Python-ffmpeg](https://github.com/jonghwanhyeon/python-ffmpeg), [AV](https://github.com/PyAV-Org/PyAV)
_computer vision and pose estimation_
* [Pydantic](https://github.com/pydantic/pydantic), [Iso639](https://github.com/janpipek/iso639-python), [PyCountry](https://github.com/pycountry/pycountry), [Nest-asyncio](https://github.com/erdewit/nest_asyncio)
_validation, and utilities_
* [Ipywidgets](https://github.com/jupyter-widgets/ipywidgets), [IpKernel](https://github.com/ipython/ipykernel), [Nbformat](https://github.com/jupyter/nbformat), [Nbss-upload](https://github.com/notebook-sharing-space/nbss-upload), [Notebook-intelligence](https://github.com/notebook-intelligence/notebook-intelligence)
_Jupyter and notebook-related tools_

We are thankful to the open-source community for enabling this project! 🙏
