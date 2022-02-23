#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GENO solver setup script"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('genosolver/_version.py').read())
setup(
    name='genosolver',
    version=__version__,
    description='GENO is a solver for non-linear optimization problems. '
                'It can solve constrained and unconstrained problems.',
    license='AGPL-3.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.geno-project.org',
    project_urls={
    'Source': 'https://github.com/slaue/genosolver/'
    },
    author='Soeren Laue',
    author_email='soeren.laue@uni-jena.de',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords=['optimization', 'machine_learning'],
    packages=find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.6',
    install_requires=['numpy>=1.17', 'packaging']
)
