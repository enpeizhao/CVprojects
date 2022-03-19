# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
u"""EXIF utilities for extracting focal length for distance calculation.

The module contains the function `get_focal_length` to extract focal length
data from image EXIF data.

Missing 35mm focal length data (usually provided by smartphone cameras) will
be calculated from the crop factor (or focal length multiplier). The crop
factor is read from a database of camera models. The function loads the
database lazily, e.g. if input data always contains the required information,
no data will ever be loaded.
"""
import csv
import os
from enum import IntEnum
from typing import Dict, Optional, Tuple
from PIL.Image import Image


DATABASE_NAME = 'camdb.csv'
_MODEL_DATABASE: Dict[str, float] = {}


class ExifTag(IntEnum):
    """EXIF tag indexes: see https://exiftool.org/TagNames/EXIF.html"""
    MODEL = 272
    ORIENTATION = 274
    FOCAL_LENGTH_IN_MM = 37386
    PIXEL_WIDTH = 40962
    PIXEL_HEIGHT = 40963
    FOCAL_LENGTH_35MM = 41989


def get_focal_length(image: Image) -> Optional[Tuple[int, int, int, int]]:
    u"""Extract focal length data from EXIF data.

    The function will try to calculate missing data (e.g. focal length in
    35mm) using a camera model database. The database will be loaded once
    it is first required.

    Args:
        image (Image): PIL image instance to get EXIF data from.

    Returns:
        (tuple) Tuple of `(focal_length_35mm, focal_length, width, height)`
        for use with `iris_depth_in_mm_from_landmarks`. `None` is returned
        if EXIF data is missing or couldn't be calculated (e.g. camera model
        missing or not in database).
    """
    exif = image.getexif()
    if ExifTag.FOCAL_LENGTH_IN_MM not in exif:
        # no focal length
        return None
    if ExifTag.PIXEL_WIDTH not in exif or ExifTag.PIXEL_HEIGHT not in exif:
        width_px, height_px = image.size
    else:
        width_px = exif[ExifTag.PIXEL_WIDTH]
        height_px = exif[ExifTag.PIXEL_HEIGHT]
        # swap width and height if orientation is rotate 90° or 270°
        if ExifTag.ORIENTATION in exif and exif[ExifTag.ORIENTATION] > 4:
            width_px, height_px = height_px, width_px
    focal_length_in_mm = exif[ExifTag.FOCAL_LENGTH_IN_MM]
    if ExifTag.FOCAL_LENGTH_35MM in exif:
        focal_length_35mm = exif[ExifTag.FOCAL_LENGTH_35MM]
    else:
        model = exif[ExifTag.MODEL] if ExifTag.MODEL in exif else None
        if model is not None and len(_MODEL_DATABASE) == 0:
            _load_database()
        if model not in _MODEL_DATABASE:
            return None
        crop_factor = _MODEL_DATABASE[model]
        focal_length_35mm = round(focal_length_in_mm * crop_factor)
    return focal_length_35mm, focal_length_in_mm, width_px, height_px


def _load_database():
    """Load camera model names and associated crop factor"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(base_path, '../data', DATABASE_NAME)
    with open(database_path, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        for model, crop_factor in reader:
            _MODEL_DATABASE[model] = float(crop_factor)
