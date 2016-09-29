# -*- coding: utf-8 -*-
"""
The :mod:`tests` package includes tests for all (in time, at least) modules
included in the OnPLS package.

Created on Mon Sep 26 22:57:17 2016

Copyright (c) 2016, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
from .tests import TestCase
from .tests import test_all

__all__ = ["TestCase", "test_all"]
