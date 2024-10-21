"""
Identifier:     transfinder/__init__.py
Name:           __init__.py
Description:    import necessary modules
Author:         Dezi Liu
Created:        2024-07-31
Modified-History:
    2024-07-31, Dezi Liu, create this function
"""

# Copyright (c) 2024 Liu Dezi
#
# This file is part of Transinder: https://github.com/LiuDezi/TransFinder.git
# MIT License
#

# TransFinder pipeline information
__version__ = "1.5.1"
__date__ = "20241020"

# import modules
from . import imgdiff
from . import utils
#from . import transdet

__all__ = ["imgdiff", "utils"]
