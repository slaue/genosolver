# -*- coding: utf-8 -*-
"""
GENO is a solver for non-linear optimization problems.
It can solve constrained and unconstrained problems.
"""
from packaging import version

__version__ = '0.1.0.6'

def check_version(new_version):
    """ Check if GENO solver should be upgraded.
    """
    if version.parse(__version__) < version.parse(new_version):
        print("A new version of the GENO solver is available. "
              "You should consider upgrading it via 'pip install --upgrade genosolver'.")
