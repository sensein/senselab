# Contributing to ```senselab```

**Pull requests** are always welcome, and we appreciate any help you give.
Note that a code of conduct applies to all spaces managed by the `senselab` project, including issues and pull requests. Please see the [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Workflow
Please use the following workflow when contributing:

0. **Install poetry and poetry-dynamic-versioning plugin for dynamic versioning**: 
  - ```pipx install poetry==1.7.1```
  - ```pipx inject poetry "poetry-dynamic-versioning[plugin]"```
1. **Create an issue**: Use GitHub to create an issuel, assign it to yourself (and any collaborators) and, if you have access, add it to the [Project board](https://github.com/orgs/sensein/projects/45).
2. **Create a branch**: Use GitHub's "Create a branch" button from the issue page to generate a branch associated with the issue.
3. **Clone the repo locally**:
   ```git clone https://github.com/sensein/senselab.git```
4. **Checkout locally**:
    - ```git fetch origin```
    - ```git checkout <branch-name>```
5. **Install all required dependencies**:
  - ```poetry run pip install iso-639```
  - ```poetry install --with dev,docs```
6. **Install pre-commit hooks**:
  ```poetry run pre-commit install```
7. **Work locally on the issue branch.**
  Note: The contributed code will be licensed under the same [license](LICENSE) as the rest of the repository. **If you did not write the code yourself, you must ensure the existing license is compatible and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.**
8. **Commit and push regularly on your dev branch.**
    - It is also OK to submit work in progress.
    - Please, write unit tests for your code and test it locally:
        ```poetry run pytest```
    - Please, document your code following [Google style guidelines](https://google.github.io/styleguide/) and the example at the end of this document.
      You can manually check the documentation automatically generated from the docstrings:
      ```poetry run pdoc```.
      This command uses ```pdoc``` to generate the documentation for you and make it accessible through a web interface.
    - If you installed the pre-commit hooks properly, some tests and checks will run, and the commit will succeed if all tests pass. If you prefer to run your tests manually, use the following commands:
      - Static type checks:
        ```poetry run mypy .```
      - Code style checks:
        ```poetry run ruff check```
        - To automatically fix issues:
          ```poetry run ruff check --fix```
      - Spell checking:
        ```poetry run codespell```
10. **Add repository secrets**: From your github web interface, add the following repository secrets: ```CODECOV_TOKEN``` (CodeCov), ```HF_TOKEN``` (HuggingFace), ```PYPI_TOKEN``` (Pypi).
11. **Submit a pull request**: Once you are done adding your new amazing functionality, submit a pull request to merge the upstream issue branch into the upstream main.
12. **Don’t worry much about point 9**: Just joking, there’s nothing there – just making sure you're paying attention!

This approach ensures that tasks, issues, and branches all have names that correspond.
It also facilitates incremental neatly scoped changes since it tends to keep the scope of individual changes narrow.

**If you would like to change this workflow, please use the current process to suggest a change to this document.**

### The biometrics-book
If you feel that the functionality you have added to senselab requires some extra explanation, or you want to share some of the knowledge you obtained during the process (e.g., you implemented an API for speaker diarization and want to write a brief explanation about what speaker diarization means and how it's generally evaluated), you can contribute to the [Biometrics-book](https://github.com/sensein/biometrics-book)!


### An example of well documented function following Google-style

````
import statistics
from typing import Dict, List

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate statistics from a list of numbers.

    Args:
        data (list of float): A list of floating-point numbers.

    Returns:
        dict: A dictionary containing the mean, median, variance, and standard deviation of the input data.

    Raises:
        ValueError: If the input data list is empty.

    Examples:
        >>> calculate_statistics([1, 2, 3, 4, 5])
        {'mean': 3.0, 'median': 3.0, 'variance': 2.0, 'std_dev': 1.4142135623730951}

        >>> calculate_statistics([2.5, 3.5, 4.5, 5.5, 6.5])
        {'mean': 4.5, 'median': 4.5, 'variance': 2.5, 'std_dev': 1.5811388300841898}

    Note:
        This function assumes the input data list is not empty. An empty list will raise a ValueError.

    Todo:
        More statistics will be implemented in the future.
    """
    if not data:
        raise ValueError("The input data list is empty.")

    mean = statistics.mean(data)
    median = statistics.median(data)
    variance = statistics.variance(data)
    std_dev = statistics.stdev(data)

    return {
        'mean': mean,
        'median': median,
        'variance': variance,
        'std_dev': std_dev
    }
````
