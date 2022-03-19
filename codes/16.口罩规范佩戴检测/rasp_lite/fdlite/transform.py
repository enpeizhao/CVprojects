# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from fdlite import ArgumentError, CoordinateRangeError, InvalidEnumError
from fdlite.types import BBox, Detection, ImageTensor, Landmark, Rect
import cv2
"""Functions for data transformations that are used by the detection models"""







"""This is our modified version of the image to tensor functions."""

def image_to_tensor(
    image: Union[PILImage, np.ndarray, str],
    roi: Optional[Rect] = None,
    output_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = False,
    output_range: Tuple[float, float] = (0., 1.),
    flip_horizontal: bool = False
) -> ImageTensor:
    """Load an image into an array and return data, image size, and padding.
    This function combines the mediapipe calculator-nodes ImageToTensor,
    ImageCropping, and ImageTransformation into one function.
    Args:
        image (Image|ndarray|str): Input image; preferably RGB, but will be
            converted if necessary; loaded from file if a string is given
        roi (Rect|None): Location within the image where to convert; can be
            `None`, in which case the entire image is converted. Rotation is
            supported.
        output_size (tuple|None): Tuple of `(width, height)` describing the
            output tensor size; defaults to ROI if `None`.
        keep_aspect_ratio (bool): `False` (default) will scale the image to
            the output size; `True` will keep the ROI aspect ratio and apply
            letterboxing.
        output_range (tuple): Tuple of `(min_val, max_val)` containing the
            minimum and maximum value of the output tensor.
            Defaults to (0, 1).
        flip_horizontal (bool): Flip the resulting image horizontally if set
            to `True`. Default: `False`
    Returns:
        (ImageTensor) Tensor data, padding for reversing letterboxing and
        original image dimensions.
    """
    img = _normalize_image(image)
    #img = image
    image_size = (img.shape[0],img.shape[1])
    #print(image_size)
    if roi is None:
        roi = Rect(0.5, 0.5, 1.0, 1.0, rotation=0.0, normalized=True)
    img_width = image_size[1]
    img_height = image_size[0]
    roi = roi.scaled(image_size)
    if output_size is None:
        output_size = (int(roi.size[0]), int(roi.size[1]))
    width, height = (roi.size if keep_aspect_ratio      # type: ignore[misc]
                     else output_size)
    src_points = np.array(roi.points())
    dst_points = np.array([(0., 0.), (width, 0.), (width, height), (0., height)])
    coeffs = _perspective_transform_coeff(src_points, dst_points)
    #roi_image = img.transform(size=(width, height), method=Image.PERSPECTIVE,
    #                         data=coeffs, resample=Image.LINEAR)
    roi_image = cv2.resize(img,output_size,coeffs,0,0)
    # free some memory - we don't need the temporary image anymore
    # if img != image:
    #     img.close()
    pad_x, pad_y = 0., 0.
    # if keep_aspect_ratio:
    #     # perform letterboxing if required
    #     out_aspect = output_size[1] / output_size[0]    # type: ignore[index]
    #     roi_aspect = roi.height / roi.width
    #     new_width, new_height = int(roi.width), int(roi.height)
    #     if out_aspect > roi_aspect:
    #         new_height = int(roi.width * out_aspect)
    #         pad_y = (1 - roi_aspect / out_aspect) / 2
    #     else:
    #         new_width = int(roi.height / out_aspect)
    #         pad_x = (1 - out_aspect / roi_aspect) / 2
    #     if new_width != int(roi.width) or new_height != int(roi.height):
    #         pad_h, pad_v = int(pad_x * new_width), int(pad_y * new_height)
    #         padding = (pad_h, pad_v)
    #         # roi_image = roi_image.transform(
    #         #      size=(new_width, new_height), method=Image.EXTENT,
    #         #      data=(-pad_h, -pad_v, new_width - pad_h, new_height - pad_v))
            
            
            
    #     #roi_image = roi_image.resize(output_size, resample=Image.BILINEAR)
    #     roi_image = cv2.resize(roi_image, output_size)
    # if flip_horizontal:
    #     roi_image = roi_image.transpose(method=Image.FLIP_LEFT_RIGHT)
    #finally, apply value range transform
    min_val, max_val = output_range
    #print(output_range)
    tensor_data = np.asarray(roi_image, dtype=np.float32)
    tensor_data *= (max_val - min_val) / 255
    tensor_data += min_val
    return ImageTensor(tensor_data,
                       padding=(pad_x, pad_y, pad_x, pad_y),
                       original_size=image_size)


"""end of function"""


def image_to_tensor_0(
    image: Union[PILImage, np.ndarray, str],
    roi: Optional[Rect] = None,
    output_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = False,
    output_range: Tuple[float, float] = (0., 1.),
    flip_horizontal: bool = False
) -> ImageTensor:
    """Load an image into an array and return data, image size, and padding.
    This function combines the mediapipe calculator-nodes ImageToTensor,
    ImageCropping, and ImageTransformation into one function.
    Args:
        image (Image|ndarray|str): Input image; preferably RGB, but will be
            converted if necessary; loaded from file if a string is given
        roi (Rect|None): Location within the image where to convert; can be
            `None`, in which case the entire image is converted. Rotation is
            supported.
        output_size (tuple|None): Tuple of `(width, height)` describing the
            output tensor size; defaults to ROI if `None`.
        keep_aspect_ratio (bool): `False` (default) will scale the image to
            the output size; `True` will keep the ROI aspect ratio and apply
            letterboxing.
        output_range (tuple): Tuple of `(min_val, max_val)` containing the
            minimum and maximum value of the output tensor.
            Defaults to (0, 1).
        flip_horizontal (bool): Flip the resulting image horizontally if set
            to `True`. Default: `False`
    Returns:
        (ImageTensor) Tensor data, padding for reversing letterboxing and
        original image dimensions.
    """
    img = _normalize_image(image)
    #img = image
    image_size = img.size
    #print(image_size)
    if roi is None:
        roi = Rect(0.5, 0.5, 1.0, 1.0, rotation=0.0, normalized=True)
    roi = roi.scaled(image_size)
    if output_size is None:
        output_size = (int(roi.size[0]), int(roi.size[1]))
    width, height = (roi.size if keep_aspect_ratio      # type: ignore[misc]
                     else output_size)
    src_points = roi.points()
    dst_points = [(0., 0.), (width, 0.), (width, height), (0., height)]
    coeffs = _perspective_transform_coeff(src_points, dst_points)
    roi_image = img.transform(size=(width, height), method=Image.PERSPECTIVE,
                              data=coeffs, resample=Image.LINEAR)
    # free some memory - we don't need the temporary image anymore
    if img != image:
        img.close()
    pad_x, pad_y = 0., 0.
    if keep_aspect_ratio:
        # perform letterboxing if required
        out_aspect = output_size[1] / output_size[0]    # type: ignore[index]
        roi_aspect = roi.height / roi.width
        new_width, new_height = int(roi.width), int(roi.height)
        if out_aspect > roi_aspect:
            new_height = int(roi.width * out_aspect)
            pad_y = (1 - roi_aspect / out_aspect) / 2
        else:
            new_width = int(roi.height / out_aspect)
            pad_x = (1 - out_aspect / roi_aspect) / 2
        if new_width != int(roi.width) or new_height != int(roi.height):
            pad_h, pad_v = int(pad_x * new_width), int(pad_y * new_height)
            roi_image = roi_image.transform(
                size=(new_width, new_height), method=Image.EXTENT,
                data=(-pad_h, -pad_v, new_width - pad_h, new_height - pad_v))
        roi_image = roi_image.resize(output_size, resample=Image.BILINEAR)
    if flip_horizontal:
        roi_image = roi_image.transpose(method=Image.FLIP_LEFT_RIGHT)
    # finally, apply value range transform
    min_val, max_val = output_range
    tensor_data = np.asarray(roi_image, dtype=np.float32)
    tensor_data *= (max_val - min_val) / 255
    tensor_data += min_val
    return ImageTensor(tensor_data,
                       padding=(pad_x, pad_y, pad_x, pad_y),
                       original_size=image_size)


def sigmoid(data: np.ndarray) -> np.ndarray:
    """Return sigmoid activation of the given data
    Args:
        data (ndarray): Numpy array containing data
    Returns:
        (ndarray) Sigmoid activation of the data with element range (0,1]
    """
    return 1 / (1 + np.exp(-data))


def detection_letterbox_removal(
    detections: Sequence[Detection],
    padding: Tuple[float, float, float, float]
) -> List[Detection]:
    """Return detections with bounding box and keypoints adjusted for padding
    Args:
        detections (list): List of detection results with relative coordinates
        padding (tuple): Tuple of (`float`,`float`,`float`,`float`) containing
            the padding value [0,1) for left, top, right and bottom sides.
    Returns:
        (list) List of detections with relative coordinates adjusted to remove
        letterboxing.
    """
    left, top, right, bottom = padding
    h_scale = 1 - (left + right)
    v_scale = 1 - (top + bottom)

    def adjust_data(detection: Detection) -> Detection:
        adjusted = (detection.data - (left, top)) / (h_scale, v_scale)
        return Detection(adjusted, detection.score)

    return [adjust_data(detection) for detection in detections]


class SizeMode(IntEnum):
    """Size mode for `bbox_to_roi`
    DEFAULT     - keep width and height as calculated
    SQUARE_LONG - make square using `max(width, height)`
    SQUARE_SHORT - make square using `min(width, height)`
    """
    DEFAULT = 0
    SQUARE_LONG = 1
    SQUARE_SHORT = 2


def bbox_to_roi(
    bbox: BBox,
    image_size: Tuple[int, int],
    rotation_keypoints: Optional[Sequence[Tuple[float, float]]] = None,
    scale: Tuple[float, float] = (1., 1.),
    size_mode: SizeMode = SizeMode.DEFAULT
) -> Rect:
    """Convert a normalized bounding box into a ROI with optional scaling and
    and rotation.
    This function combines parts of DetectionsToRect and RectTransformation
    MediaPipe nodes.
    Args:
        bbox (bbox): Normalized bounding box to convert.
        image_size (tuple): Image size for the bounding box as a tuple
            of `(image_width, image_height)`.
        rotation_keypoints (list|None): Optional list of keypoints to get the
            target rotation from; expected format: `[(x1, y1), (x2, y2)]`
        scale (tuple): Tuple of `(scale_x, scale_y)` that determines the
            scaling of the requested ROI.
        size_mode (SizeMode): Determines the way the ROI dimensions should be
            determined. Default keeps the bounding box proportions as-is,
            while the other modes result in a square with a length matching
            either the shorter or longer side.
    Returns:
        (Rect) Normalized and possibly rotated ROI rectangle.
    Raises:
        CoordinateRangeError: bbox is not in normalised coordinates (0 to 1)
        InvalidEnumError: `size_mode` contains an unsupported value
    """
    if not bbox.normalized:
        raise CoordinateRangeError('bbox must be normalized')
    PI = np.math.pi
    TWO_PI = 2 * PI
    # select ROI dimensions
    width, height = _select_roi_size(bbox, image_size, size_mode)
    scale_x, scale_y = scale
    # calculate ROI size and -centre
    width, height = width * scale_x, height * scale_y
    cx, cy = bbox.xmin + bbox.width / 2, bbox.ymin + bbox.height / 2
    # calculate rotation of required
    if rotation_keypoints is None or len(rotation_keypoints) < 2:
        return Rect(cx, cy, width, height, rotation=0., normalized=True)
    x0, y0 = rotation_keypoints[0]
    x1, y1 = rotation_keypoints[1]
    angle = -np.math.atan2(y0 - y1, x1 - x0)
    # normalise to [0, 2*PI]
    rotation = angle - TWO_PI * np.math.floor((angle + PI) / TWO_PI)
    return Rect(cx, cy, width, height, rotation, normalized=True)


def bbox_from_landmarks(landmarks: Sequence[Landmark]) -> BBox:
    """Return the bounding box that encloses all landmarks in a given list.
    This function combines the MediaPipe nodes LandmarksToDetectionCalculator
    and DetectionToRectCalculator.
    Args:
        landmarks (list): List of landmark detection results. Must contain ar
            least two items.
    Returns:
        (BBox) Bounding box that contains all points defined by the landmarks.
    Raises:
        ArgumentError: `landmarks` contains less than two (2) items
    """
    if len(landmarks) < 2:
        raise ArgumentError('landmarks must contain at least 2 items')
    xmin, ymin = 999999., 999999.
    xmax, ymax = -999999., -999999.
    for landmark in landmarks:
        x, y = landmark.x, landmark.y
        xmin, ymin = min(xmin, x), min(ymin, y)
        xmax, ymax = max(xmax, x), max(ymax, y)
    return BBox(xmin, ymin, xmax, ymax)


def project_landmarks(
    data: Union[Sequence[Landmark], np.ndarray],
    *,
    tensor_size: Tuple[int, int],
    image_size: Tuple[int, int],
    padding: Tuple[float, float, float, float],
    roi: Optional[Rect],
    flip_horizontal: bool = False
) -> List[Landmark]:
    """Transform landmarks or raw detection results from tensor coordinates
    into normalized image coordinates, removing letterboxing if required.
    Args:
        data (list|ndarray): List of landmarks or numpy array with number of
            elements divisible by 3.
        tensor_size (tuple): Tuple of `(width, height)` denoting the input
            tensor size.
        image_size (tuple): Tuple of `(width, height)` denoting the image
            size.
        padding (tuple): Tuple of `(pad_left, pad_top, pad_right, pad_bottom)`
            denoting padding from letterboxing.
        roi (Rect|None): Optional ROI from which the input data was taken.
        flip_horizontal (bool): Flip the image from left to right if `True`
    Returns:
        (list) List of normalized landmarks projected into image space.
    """
    # normalize input type
    if not isinstance(data, np.ndarray):
        points = np.array([(pt.x, pt.y, pt.z) for pt in data], dtype='float32')
    else:
        points = data.reshape(-1, 3)
    # normalize to tensor coordinates
    width, height = tensor_size
    points /= (width, height, width)
    # flip left to right if requested
    if flip_horizontal:
        points[:, 0] *= -1
        points[:, 0] += 1
    # letterbox removal if required
    if any(padding):
        left, top, right, bottom = padding
        h_scale = 1 - (left + right)
        v_scale = 1 - (top + bottom)
        points -= (left, top, 0.)
        points /= (h_scale, v_scale, h_scale)
    # convert to landmarks if coordinate system doesn't change
    if roi is None:
        return [Landmark(x, y, z) for (x, y, z) in points]
    # coordinate system transformation from ROI- to image space
    norm_roi = roi.scaled(image_size, normalize=True)
    sin, cos = np.math.sin(roi.rotation), np.math.cos(roi.rotation)
    matrix = np.array([[cos, sin, 0.], [-sin, cos, 0.], [1., 1., 1.]])
    points -= (0.5, 0.5, 0.0)
    rotated = np.matmul(points * (1, 1, 0), matrix)
    points *= (0, 0, 1)
    points += rotated
    points *= (norm_roi.width, norm_roi.height, norm_roi.width)
    points += (norm_roi.x_center, norm_roi.y_center, 0.0)
    return [Landmark(x, y, z) for (x, y, z) in points]


def _perspective_transform_coeff(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> np.ndarray:
    """Calculate coefficients for a perspective transform given source- and
    target points. Note: argument order is reversed for more intuitive
    usage.
    Reference:
    https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/
    """
    matrix = []
    for (x, y), (X, Y) in zip(dst_points, src_points):
        matrix.extend([
            [x, y, 1., 0., 0., 0., -X*x, -X*y],
            [0., 0., 0., x, y, 1., -Y*x, -Y*y]
        ])
    A = np.array(matrix, dtype=np.float32)
    B = np.array(src_points, dtype=np.float32).reshape(8)
    return np.linalg.solve(A, B)


def _normalize_image(image: Union[PILImage, np.ndarray, str]) -> PILImage:
    """Return PIL Image instance in RGB-mode from input"""
    if isinstance(image, PILImage) and image.mode != 'RGB':
        return image.convert(mode='RGB')
    if isinstance(image, np.ndarray):
        #return Image.fromarray(image, mode='RGB')
        return image
    if not isinstance(image, PILImage):
        return Image.open(image)
    return image


def _select_roi_size(
    bbox: BBox,
    image_size: Tuple[int, int],
    size_mode: SizeMode
) -> Tuple[float, float]:
    """Return the size of an ROI based on bounding box, image size and mode"""
    abs_box = bbox.absolute(image_size)
    width, height = abs_box.width, abs_box.height
    image_width, image_height = image_size
    if size_mode == SizeMode.SQUARE_LONG:
        long_size = max(width, height)
        width, height = long_size / image_width, long_size / image_height
    elif size_mode == SizeMode.SQUARE_SHORT:
        short_side = min(width, height)
        width, height = short_side / image_width, short_side / image_height
    elif size_mode != SizeMode.DEFAULT:
        raise InvalidEnumError(f'unsupported size_mode: {size_mode}')
    return width, height