# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Union
"""Types used throughout the library"""


@dataclass
class ImageTensor:
    """Tensor data obtained from an image with optional letterboxing.
    The original image size is kept track of.

    The data may contain an extra dimension for batching (the default).
    """
    tensor_data: np.ndarray
    padding: Tuple[float, float, float, float]
    original_size: Tuple[int, int]


@dataclass
class Rect:
    """A rotated rectangle

    Rotation is given in radians (clockwise).
    Normalized indicates whether properties are relative to image size
    (i.e. value range is [0,1]).
    """
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float
    normalized: bool

    @property
    def size(self) -> Union[Tuple[float, float], Tuple[int, int]]:
        """Tuple of `(width, height)`"""
        w, h = self.width, self.height
        return (w, h) if self.normalized else (int(w), int(h))

    def scaled(
        self, size: Tuple[float, float], normalize: bool = False
    ) -> 'Rect':
        """Return a scaled version or self, if not normalized."""
        if self.normalized == normalize:
            return self
        sx, sy = size
        if normalize:
            sx, sy = 1 / sx, 1 / sy
        return Rect(self.x_center * sx, self.y_center * sy,
                    self.width * sx, self.height * sy,
                    self.rotation, normalized=False)

    def points(self) -> np.ndarray:
        """Return the corners of the box as a list of tuples `[(x, y), ...]`
        """
        x, y = self.x_center, self.y_center
        w, h = self.width / 2, self.height / 2
        pts = [(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)]
        if self.rotation == 0:
            return pts
        s, c = np.math.sin(self.rotation), np.math.cos(self.rotation)
        t = np.array(pts) - (x, y)
        r = np.array([[c, s], [-s, c]])
        return np.matmul(t, r) + (x, y)


@dataclass
class BBox:
    """A non-rotated bounding box.

    The bounds can be relative to image size (normalized, range [0,1]) or
    absolute (i.e. pixel-) coordinates.
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Upper-left and bottom-right as a tuple"""
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def width(self) -> float:
        """Width of the box"""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Height of the box"""
        return self.ymax - self.ymin

    @property
    def empty(self) -> bool:
        """True if the box is empty"""
        return self.width <= 0 or self.height <= 0

    @property
    def normalized(self) -> bool:
        """True if the box contains normalized coordinates"""
        return self.xmin >= -1 and self.xmax < 2 and self.ymin >= -1

    @property
    def area(self) -> float:
        """Area of the bounding box, 0 if empty"""
        return self.width * self.height if not self.empty else 0

    def intersect(self, other: 'BBox') -> Optional['BBox']:
        """Return the intersection with another (non-rotated) bounding box

        Args:
            other (BBox): Bounding box to intersect with

        Returns:
            (BBox) Intersection between the two bounding boxes; `None` if the
            boxes are disjoint.
        """
        xmin, ymin = max(self.xmin, other.xmin), max(self.ymin, other.ymin)
        xmax, ymax = min(self.xmax, other.xmax), min(self.ymax, other.ymax)
        if xmin < xmax and ymin < ymax:
            return BBox(xmin, ymin, xmax, ymax)
        else:
            return None

    def scale(self, size: Tuple[float, float]) -> 'BBox':
        """Scale the bounding box"""
        sx, sy = size
        xmin, ymin = self.xmin * sx, self.ymin * sy
        xmax, ymax = self.xmax * sx, self.ymax * sy
        return BBox(xmin, ymin, xmax, ymax)

    def absolute(self, size: Tuple[int, int]) -> 'BBox':
        """Return the box in absolute coordinates (if normalized)

        Args:
            size (tuple): Tuple of `(image_width, image_height)` that denotes
                the image dimensions. Ignored if the box is not normalized.

        Returns:
            (BBox) Bounding box in absolute pixel coordinates.
        """
        if not self.normalized:
            return self
        return self.scale(size)


@dataclass
class Landmark:
    """An object landmark (3d point) detected by a model"""
    x: float
    y: float
    z: float


class Detection(object):
    """An object detection made by a model.

    A detection consists of a bounding box and zero or more 2d keypoints.
    Keypoints can be accessed directly via indexing or iterating a detection:

    ```
        detection = ...
        # loop through keypoints
        for keypoint in detection:
            print(keypoint)
        # access a keypoint by index
        nosetip = detection[3]
    ```
    """
    def __init__(self, data: np.ndarray, score: float) -> None:
        """Initialize a detection from data points.

        Args:
            data (ndarray): Array of `[xmin, ymin, xmax, ymax, ...]` followed
                by zero or more x and y coordinates.

            score (float): Confidence score of the detection; range [0, 1].
        """
        self.data = data.reshape(-1, 2)
        self.score = score

    def __len__(self) -> int:
        """Number of keypoints"""
        return len(self.data) - 2

    def __getitem__(self, key: int) -> Tuple[float, float]:
        """Keypoint by index"""
        x, y = self.data[key + 2]
        return x, y

    def __iter__(self):
        """Keypoints iterator"""
        return iter(self.data[2:])

    @property
    def bbox(self) -> BBox:
        """The bounding box of this detection."""
        xmin, ymin = self.data[0]
        xmax, ymax = self.data[1]
        return BBox(xmin, ymin, xmax, ymax)

    def scaled(self, factor: Union[Tuple[float, float], float]) -> 'Detection':
        """Return a scaled version of the bounding box and keypoints"""
        return Detection(self.data * factor, self.score)
