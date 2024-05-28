## ```senselab``` pull request guidelines
Pull requests are always welcome, and we appreciate any help you give. Note that a code of conduct applies to all spaces managed by the ```senselab``` project, including issues and pull requests. Please see the [Code of Conduct](CODE_OF_CONDUCT.md) for details.


### Workflow
Please use the following workflow when contributing:
1. Create an issue, assign it to yourself (and any collaborators), assign labels, and (if you have access) add it to the [Project board](https://github.com/orgs/sensein/projects/45)
2. Use GitHub's "Create a branch" button from the issue page to generate a branch associated with the issue
3. Fork the issue branch
4. Work locally on the forked issue branch
5. Add the following repository secrets: ```CODECOV_TOKEN``` (CodeCov), ```HF_TOKEN``` (HuggingFace), ```PYPI_TOKEN``` (Pypi).
6. Submit a pull request to merge the forked issue branch into the upstream issue branch.
   **It is also OK to submit work in progress**.
7. After the pull request from 6 is merged, submit a pull request to merge the upstream issue branch into the upstream main

This approach ensures that tasks, issues, and branches all have names that correspond. It also facilitates incremental neatly scoped changes since it tends to keep the scope of individual changes narrow.

**If you would like to change this workflow, please use the current process to suggest a change to this document.**


### Submitting a pull request
When submitting a pull request, we ask you to check the following:

1. **Unit tests**, **code style**, **docstring** are in order.
   - Execute all static type checks by running the command:
     ```poetry run mypy .```
     This command uses ``mypy``` to run all typing checking and points you at those lines where it identifies some issues.
   - Execute all unit tests to verify functionality through the following command:
     ```poetry run pytest```
      This command uses ```pytest``` to run tests, helping to identify any issues early.
   - Check your code for style guidelines and common errors by running:
     ```poetry run ruff check```
      This command utilizes ```ruff```, a code style checker, to ensure your code adheres to the project's style guidelines.
     By running ```poetry run ruff check --fix```, ruff will automatically fix issues for you.
   - Check that the documentation automatically generated from the docstrings looks in order with the following command:
     ```poetry run pdoc```
     This command uses ```pdoc``` to generate the documentation for you and make it accessible through a web interface. 

2. **See the Continuous Integration** for up-to-date information on the current tests, code style, and other requirements.

3. The contributed code will be licensed under the same [license](LICENSE) as the rest of the repository. **If you did not write the code yourself, you must ensure the existing license is compatible** and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.


### The biometrics-book
If you feel that the functionality you have added to ```senselab``` requires some extra explanation and/or you want to share some of the knowledge you obtained during the process (e.g., you implemented an API for speaker diarization and want to write a brief explanation about what does speaker diarization mean and how it's generally evaluated), you can contribute to the [Biometrics-book](https://github.com/sensein/biometrics-book)! 
