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





