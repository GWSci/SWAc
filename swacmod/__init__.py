#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWAcMod init."""

from _version import version
from _commit_id import commit_id

__version__ = ".".join([str(i) for i in version])
__commit_id__ = commit_id
__authors__ = ["Alastair Black", "Marco Lagi"]
__author_email__ = "a.black@GWScience.co.uk"
__copyright__ = "Copyright (C) 2016 Groundwater Science Ltd."
__license__ = "GNU GENERAL PUBLIC LICENSE"
__url__ = "https://github.com/AlastairBlack/swacmod"
