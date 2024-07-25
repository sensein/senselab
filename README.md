# The ```senselab``` repo

[![Build](https://github.com/sensein/senselab/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sensein/senselab/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sensein/senselab/graph/badge.svg?token=9S8WY128PO)](https://codecov.io/gh/sensein/senselab)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/senselab.svg)](https://pypi.org/project/senselab/)
[![Python Version](https://img.shields.io/pypi/pyversions/senselab)](https://pypi.org/project/senselab)
[![License](https://img.shields.io/pypi/l/senselab)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/senselab)

Welcome to the ```senselab``` repo! This is a Python package for streamlining the processing and analysis of behavioral data, such as voice and speech patterns, with robust and reproducible methodologies.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Installation
Install this package via:

```sh
pip install senselab
```

Or get the newest development version via:

```sh
pip install git+https://github.com/sensein/senselab.git
```

## Quick start
```Python
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios

audio1 = Audio.from_filepath('path_to_audio_file.wav')

print("The original audio has a sampling rate of {} Hz.".format(audio1.sampling_rate))
[audio1] = resample_audios([audio1], resample_rate=16000)
print("The resampled audio has a sampling rate of {} Hz.".format(audio1.sampling_rate))
```

For more detailed information, check out our [**Getting Started Tutorial**](https://github.com/sensein/senselab/blob/main/tutorials/getting_started.ipynb).


## Why should I use senselab?
- **Modular Design**: Easily integrate or use standalone transformations for flexible data manipulation.
- **Pre-built Pipelines**: Access pre-configured pipelines to reduce setup time and effort.
- **Reproducibility**: Ensure consistent and verifiable results with fixed seeds and version-controlled steps.
- **Easy Integration**: Seamlessly fit into existing workflows with minimal configuration.
- **Extensible**: Modify and contribute custom transformations and pipelines to meet specific research needs.
- **Comprehensive Documentation**: Detailed guides, examples, and documentation for all features and modules.
- **Performance Optimized**: Efficiently process large datasets with optimized code and algorithms.
- **Interactive Examples**: Jupyter notebooks provide practical examples for deriving insights from real-world datasets.

## Contributing
Please see [**CONTRIBUTING.md**](https://github.com/sensein/senselab/blob/main/CONTRIBUTING.md) before contributing.

To find out what's currently in progress, please check the [**Project Board**](https://github.com/orgs/sensein/projects/45).
