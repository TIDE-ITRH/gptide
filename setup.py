#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

setup(name='gptide',
    version='0.3.1',
    url='https://github.com/TIDE-ITRH/gptide',
    description='Gaussian Process regression tools for oceanography and engineering',
    author='Lachlan Astfalck; Matt Rayson; Andrew Zulberti',
    author_email='andrew.zulberti@uwa.edu.au',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'xarray','emcee'],
    license='MIT',
    include_package_data=True,
    distclass=BinaryDistribution,
)
