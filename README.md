# The Sensein's python package template repo

[![Build](YOUR_PACKAGE_REPO/actions/workflows/test.yaml/badge.svg?branch=main)](YOUR_PACKAGE_REPO/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/YOUR_GITHUB_NICKNAME/YOUR_PACKAGE_NAME/branch/main/graph/badge.svg?token=CODECOV_GRAPHICS_TOKEN)](https://codecov.io/gh/YOUR_GITHUB_NICKNAME/YOUR_PACKAGE_NAME)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/YOUR_PACKAGE_NAME.svg)](https://pypi.org/project/YOUR_PACKAGE_NAME/)
[![Python Version](https://img.shields.io/pypi/pyversions/YOUR_PACKAGE_NAME)](https://pypi.org/project/YOUR_PACKAGE_NAME)
[![License](https://img.shields.io/pypi/l/YOUR_PACKAGE_NAME)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://YOUR_GITHUB_NICKNAME.github.io/YOUR_PACKAGE_NAME)



Welcome to the [Sensein](https://sensein.group/) python template repo! 
This template is here to help you kick off your projects with a clean and efficient setup. 
It's inspired by the [python template by the Child Mind Institute](https://github.com/childmindresearch/template-python-repository) (thanks you!). Our version diverges in its setup process and in both the variety and quantity of automated features included.

Just follow these steps, and you'll be on your way:
1. **Choose a unique package name:** First off, you need a cool name for your package. To make sure it's not already taken on PyPI, head over to `https://pypi.org/project/YOUR_PACKAGE_NAME/`. If you get a "Not Found" page, you're likely good to go!

2. **Use this template:** Go to the GitHub page for this template. You'll find a "Use this template" button on the top right. Click it to start setting up your project with the template's structure.

3. **Configure your project:** During the setup, you'll specify some basics like your project's name and whether it's public or private. You will be able to change this info in the future, no pressure!

4. **Add GitHub Secrets:** For automated processes, add these secrets to your GitHub repo:

- `PYPI_TOKEN`: Your token for PyPI, allowing GitHub Actions to publish your package.
- `AUTO_ORG_TOKEN`: A token for automated organization actions (this is useful for using [auto](https://github.com/intuit/auto) for automatic changelog generation).
- `CODECOV_TOKEN`: Your [Codecov](https://about.codecov.io/) token for reporting code coverage.

To obtain these tokens:
- For `PYPI_TOKEN`, log in to your PyPI account, go to your account settings, and create an API token.
- `AUTO_ORG_TOKEN` is a personal access token from GitHub, used for actions requiring organization-level permissions. Generate one in your GitHub settings under Developer settings > Personal access tokens. Alternatively, ask the admin of your organization to provide one.
- For `CODECOV_TOKEN`, sign up or log in to Codecov, add your repository, and you'll be provided with a token. Alternatively, ask the admin of your organization to do so.

To add these tokens:
Go to your repository on GitHub, click on "Settings" > "Secrets" > "Actions", then click on "New repository secret". Name your secret (e.g., `PYPI_TOKEN`) and paste the token value. Repeat this for each token.

5. **Clone the repo:** Once your repository is set up, clone it to your local machine.

6. **Replace placeholders with custom values**: Please, run `python template_setup.py --package-name YOUR_PACKAGE_NAME --package-repo-without-git-extension YOUR_PACKAGE_REPO --github-nickname YOUR_GITHUB_NICKNAME --codecov-token CODECOV_GRAPHICS_TOKEN --email YOUR_EMAIL`. For example, `python-package-template % python template_setup.py --package-name pipepal --package-repo-without-git-extension https://github.com/fabiocat93/pipepal --github-nickname fabiocat93 --codecov-token IQR1RCYMAA --email fabiocat@mit.edu`. This will replace some placeholders in the entire directory (including folder names, file names, file content) with your custom info:
- YOUR_PACKAGE_NAME (e.g., `pipepal`)
- YOUR_PACKAGE_REPO (e.g., `https://github.com/sensein/pipepal`)
- YOUR_GITHUB_NICKNAME (e.g., `sensein`)
- CODECOV_GRAPHICS_TOKEN (e.g.,`ABC0DEFGHI`)
- YOUR_EMAIL (e.g., `sensein@mit.edu`)
It will also enable GitHub custom automation and delete the `template_setup.py` (you won't need that anymore).

7. **Adjust `pyproject.toml`:** Please, double-check `pyproject.toml` and update it with some custom info, if needed (i.e., `description`, `authors`, `maintainers`, `description`, `homepage`, `repository`, `keywords`, and `classifiers`). No need to touch `version`. This will be automatically handled by the package.

8. **Update README.md:** Replace the content of this README.md with information specific to your project.

9. **Install poetry:** Poetry is a fantastic tool for managing dependencies and packaging. If you haven't installed it yet, check out their [documentation](https://python-poetry.org/docs/) for guidance. It's pretty straightforward.

10. **Verify poetry setup:** Run `poetry --help` to ensure everything is set up correctly.

11. **Install dependencies:** Get all your project's dependencies in place by running `poetry install --with dev`.

12. **Secure your package name:** Even if you're not quite ready to publish, consider securing your package name on PyPI. You can do this by publishing a dummy version (0.0.1) of your package with `poetry publish --build`.

13. **Commit and push:** Now's the time to add (e.g., `git add .`) and commit (e.g., `git commit -m "here goes a wonderful message"`) your changes. Consider adding a tag for your initial version (recommended), like `git tag 0.0.1`, then push it all to GitHub with `git push --tags` and `git push origin main`.

14. **Check GitHub actions:** If your push was successful, it'll trigger some GitHub Actions like code quality checks and tests. Make sure everything passes!

15. **Work in dev branch:** For future changes, create a `dev` branch and make your updates there. Use pull requests to merge these changes into the main branch.

16. **Releasing new versions:** If you want to release a new version of your package, add a "release" label to your pull request. This will trigger all the necessary actions to update the version tag, create a changelog, release the new version, and even create/update your package documentation.

17. **Set up API documentation:** After your first successful pull request, set up your API documentation website. Go to your repository's settings, find the GitHub Pages section, and select `docs` as the source. You'll get a link to your API docs.

18. **[Bonus] Customize issue and pull request remplates:** Optionally, you can customize your issue and pull request remplates from the `.github` folder.

That's it! With these steps, you're well on your way to creating an awesome Python package. Keep up the great work, and **happy coding**!


# Boilerplate for ```YOUR_PACKAGE_NAME``` readme.md
Welcome to the ```YOUR_PACKAGE_NAME``` repo! This is a Python package for doing incredible stuff.

**Caution:**: this package is still under development and may change rapidly over the next few weeks.

## Features
- A few
- Cool
- Things
- These may include a wonderful CLI interface.

## Installation
Install this package via :

```sh
pip install YOUR_PACKAGE_NAME
```

Or get the newest development version via:

```sh
pip install git+YOUR_PACKAGE_REPO.git
```

## Quick start
```Python
from YOUR_PACKAGE_NAME.app import hello_world

hello_world()
```

## To do:
- [ ] A
- [ ] lot