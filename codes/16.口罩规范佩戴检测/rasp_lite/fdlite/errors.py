# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
"""This module contains all custom excpetions used by the library"""


class InvalidEnumError(Exception):
    """Raised when a function was called with an invalid Enum value"""


class ModelDataError(Exception):
    """Raised when a model returns data that is incompatible"""


class CoordinateRangeError(Exception):
    """Raised when coordinates are expected to be in a different range"""


class ArgumentError(Exception):
    """Raised when an argument is of the wrong type or malformed"""


class MissingExifDataError(Exception):
    """Raised if required EXIF data is missing from an image"""
