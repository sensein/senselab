[![Build](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml/badge.svg)](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml)
[![codecov](https://codecov.io/gh/sensein/senselab/graph/badge.svg?token=9S8WY128PO)](https://codecov.io/gh/sensein/senselab)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/senselab.svg)](https://pypi.org/project/senselab/)
[![Python Version](https://img.shields.io/pypi/pyversions/senselab)](https://pypi.org/project/senselab)
[![License](https://img.shields.io/pypi/l/senselab)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/senselab)

Welcome to ```senselab```! This is a Python package for streamlining the processing and analysis of behavioral data, such as voice and speech patterns, with robust and reproducible methodologies.

### Why should I use senselab?
- **Modular Design**: Easily integrate or use standalone transformations for flexible data manipulation.
- **Pre-built Pipelines**: Access pre-configured pipelines to reduce setup time and effort.
- **Reproducibility**: Ensure consistent and verifiable results with fixed seeds and version-controlled steps.
- **Easy Integration**: Seamlessly fit into existing workflows with minimal configuration.
- **Extensible**: Modify and contribute custom transformations and pipelines to meet specific research needs.
- **Comprehensive Documentation**: Detailed guides, examples, and documentation for all features and modules.
- **Performance Optimized**: Efficiently process large datasets with optimized code and algorithms.
- **Interactive Examples**: Jupyter notebooks provide practical examples for deriving insights from real-world datasets.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

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

## Installation
Install this package via:

```sh
pip install senselab['all']
```

Or get the newest development version via:

```sh
pip install git+https://github.com/sensein/senselab.git
```

If you want to install only audio dependencies, you do:
```sh
pip install senselab['audio']
```
To install video and text extras, please do:
```sh
pip install senselab['video,text']
```

## Quick start
```Python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios

audio1 = Audio.from_filepath('path_to_audio_file.wav')

print("The original audio has a sampling rate of {} Hz.".format(audio1.sampling_rate))
[audio1] = resample_audios([audio1], resample_rate=16000)
print("The resampled audio has a sampling rate of {} Hz.".format(audio1.sampling_rate))
```

For more detailed information, check out our [**Getting Started Tutorial**](https://github.com/sensein/senselab/blob/main/tutorials/audio/getting_started.ipynb).


## Contributing
We welcome contributions from the community! Before getting started, please review our [**CONTRIBUTING.md**](https://github.com/sensein/senselab/blob/main/CONTRIBUTING.md).

## Acknowledgments
`senselab` is mostly supported by the following organizations and initiatives:
- McGovern Institute ICON Fellowship
- NIH Bridge2AI Precision Public Health (OT2OD032720)
- Child Mind Institute
- ReadNet Project
- Chris and Lann Woehrle Psychiatric Fund
