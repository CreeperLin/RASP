#!/usr/bin/env python
import os

from setuptools import setup, find_packages

root_dir = os.path.abspath(os.path.dirname(__file__))

readme = open(os.path.join(root_dir, 'README.md')).read()

requirements = [name.rstrip() for name in open(os.path.join(root_dir, 'requirements.txt')).readlines()]

VERSION = '0.0.1'

setup(
    name = 'rasp',
    version = VERSION,
    author = 'CreeperLin',
    author_email = 'linyunfeng@sjtu.edu.cn',
    url = 'https://github.com/CreeperLin/RASP',
    description = 'Runtime Analyzer and Statistical Profiler for NN',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    license = license,
    packages = find_packages(exclude=('test')),
    install_requires = requirements,
    classifiers = [
        'Programming Language :: Python :: 3',
    ],
)
