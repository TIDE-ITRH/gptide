[![Documentation status](https://readthedocs.org/projects/gptide/badge/?version=latest)](https://gptide.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/gptide.svg)](https://badge.fury.io/py/gptide)
[![Downloads](https://static.pepy.tech/personalized-badge/gptide?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/gptide)

<img width="500" height="300" title="logo" alt="Alt text" src="/docs/_static/gptide_thegoat.png">


# gptide

Gaussian Process regression toolkit for Transformation of Infrastructure through Digitial Engineering applications.

Gaussian Process regression (also called *Optimal Interpolation* or *Kriging*) is useful for fitting a continuous surface to sparse observations, i.e. making predictions. Its main use in environmental sciences, like oceanography, is for spatio-temporal modelling. This package provides a fairly simple API for making predictions AND for estimating kernel hyper-parameters. The hyper-parameter estimation has two main functions: one for Bayesians, one for frequentists. You choose.

Please see the [examples](https://gptide.readthedocs.io/en/latest/examples.html) for particular use cases.

Note that there are many other Gaussian Process packages on the world wide web - this package is yet another one. The selling point of this package is that the object is fairly straightforward and the kernel building is all done with functions, not abstract classes. The intention is to use this package as both a teaching and research tool.

## Documentation

Documentation is available on  [read the docs](https://gptide.readthedocs.io/en/latest/).

## Installation

### pip

`pip install gptide` 

### To install a local development version

`pip install -e ./`

### To install latest from github

`pip install git+https://github.com/mrayson/tide-itrh/gptide.git`

## Quick Usage




## Contributing

Standard workflow for collaborators, branch → PR → review → merge → release:

1. **Branch** off `main` for your change: `git checkout -b <short-description>`.
2. **Edit and commit** on that branch.
3. **Open a PR** back into `main` (`gh pr create --fill`). This triggers the test suite (`.github/workflows/tests.yml`) across Python 3.10-3.12.
4. **Review**: get at least one review before merging -- from a collaborator, or an AI review pass (e.g. Claude Code's `/code-review`) if a human reviewer isn't available that day. 
5. **Merge** once CI is green and the PR is approved (squash merge keeps history clean, but a regular merge is fine too).
6. **Tag** new versions after merge, e.g., `git tag vX.Y.Z` and `git push origin vX.Y.Z`. Tags don't automatically publish so we can tag updates for collaborator records. 
7. **Release** -- only when the change is worth shipping to users, not on every merge: create a GitHub Release from `main` (`gh release create vX.Y.Z`, following [semantic versioning](https://semver.org/)). The version is derived automatically from the tag via `setuptools_scm`, so there's no manual version bump. Publishing a release triggers `.github/workflows/publish.yml`, which re-runs the tests and then publishes to PyPI via trusted publishing.

