# The ```pipepal``` repo

[![Build](https://github.com/sensein/pipepal/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sensein/pipepal/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sensein/pipepal/branch/main/graph/badge.svg?token=MFU1LM80ET)](https://codecov.io/gh/sensein/pipepal)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/pipepal.svg)](https://pypi.org/project/pipepal/)
[![Python Version](https://img.shields.io/pypi/pyversions/pipepal)](https://pypi.org/project/pipepal)
[![License](https://img.shields.io/pypi/l/pipepal)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.github.io/pipepal)

Welcome to the ```pipepal``` repo! This is a Python package for streamlining the processing and analysis of behavioral data, such as voice and speech patterns, with robust and reproducible methodologies. 

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Features
- **Modular design**: Utilize a variety of task-specific transformations that can be easily integrated or used standalone, allowing for flexible data manipulation and analysis strategies.

- **Pre-built pipelines**: Access pre-configured pipelines combining multiple transformations tailored for common analysis tasks, which help in reducing setup time and effort.

- **Reproducibility**: Ensures consistent outputs through the use of fixed seeds and version-controlled processing steps, making your results verifiable and easily comparable.

- **Easy integration**: Designed to fit into existing workflows with minimal configuration, `pipepal` can be used alongside other data analysis tools and frameworks seamlessly.

- **Extensible**: Open to modifications and contributions, the package can be expanded with custom transformations and pipelines to meet specific research needs. <u>Do you want to contribute? Please, reach out!</u>

- **Comprehensive documentation**: Comes with detailed documentation for all features and modules, including examples and guides on how to extend the package for other types of behavioral data analysis.

- **Performance Optimized**: Efficiently processes large datasets with optimized code and algorithms, ensuring quick turnaround times even for complex analyses.

- **Interactive Examples**: Includes Jupyter notebooks that provide practical examples of how `pipepal` can be implemented to derive insights from real-world data sets.

Whether you're researching speech disorders, analyzing customer service calls, or studying communication patterns, `pipepal` provides the tools and flexibility needed to extract meaningful conclusions from your data.


## Installation
Install this package via:

```sh
pip install pipepal
```

Or get the newest development version via:

```sh
pip install git+https://github.com/sensein/pipepal.git
```

## Quick start
```Python
from pipepal.app import hello_world

hello_world()
```

## To do:
- [ ] Integrating more audio tasks and moving functions from b2aiprep package:
    - [ ] data_augmentation 
    - [ ] data_representation
    - [x] example_task
    - [x] input_output
    - [ ] raw_signal_processing
    - [ ] speaker_diarization
    - [ ] speech emotion recognition
    - [ ] speech enhancement
    - [ ] speech_to_text
    - [ ] text_to_speech
    - [ ] voice conversion
- [ ] Integrating more video tasks:
    - [x] input_output

- [ ] Preparing some pipelines with pydra
- [ ] Populating the CLI
