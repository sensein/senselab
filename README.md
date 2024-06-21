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

## Features
- **Modular design**: Utilize a variety of task-specific transformations that can be easily integrated or used standalone, allowing for flexible data manipulation and analysis strategies.

- **Pre-built pipelines**: Access pre-configured pipelines combining multiple transformations tailored for common analysis tasks, which help in reducing setup time and effort.

- **Reproducibility**: Ensures consistent outputs through the use of fixed seeds and version-controlled processing steps, making your results verifiable and easily comparable.

- **Easy integration**: Designed to fit into existing workflows with minimal configuration, `senselab` can be used alongside other data analysis tools and frameworks seamlessly.

- **Extensible**: Open to modifications and contributions, the package can be expanded with custom transformations and pipelines to meet specific research needs. <u>Do you want to contribute? Please, reach out!</u>

- **Comprehensive documentation**: Comes with detailed documentation for all features and modules, including examples and guides on how to extend the package for other types of behavioral data analysis.

- **Performance Optimized**: Efficiently processes large datasets with optimized code and algorithms, ensuring quick turnaround times even for complex analyses.

- **Interactive Examples**: Includes Jupyter notebooks that provide practical examples of how `senselab` can be implemented to derive insights from real-world data sets.

Whether you're researching speech disorders, analyzing customer service calls, or studying communication patterns, `senselab` provides the tools and flexibility needed to extract meaningful conclusions from your data.


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
from senselab.app import hello_world

hello_world()
```

## Contributing
Please see [CONTRIBUTING.md](CONTRIBUTING.md) before contributing.

## To do
Please see the [Project Board](https://github.com/orgs/sensein/projects/45).
