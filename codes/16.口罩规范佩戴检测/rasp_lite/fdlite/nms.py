# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
import numpy as np
from typing import List, Optional, Tuple
from fdlite.types import BBox, Detection
"""Implementation of non-maximum-suppression (NMS) for detections."""


def non_maximum_suppression(
    detections: List[Detection],
    min_suppression_threshold: float,
    min_score: Optional[float],
    weighted: bool = False
) -> List[Detection]:
    """Return only significant detections.

    Args:
        detections (list): List of detections.

        min_suppression_threshold (float): Discard detections whose similarity
            is above this threshold value

        min_score (float): Minimum score of valid detections.

    Returns:
        (list) List of sufficiently relevant detections
    """
    scores = [detection.score for detection in detections]
    indexed_scores = [(n, score) for n, score in enumerate(scores)]
    indexed_scores = sorted(indexed_scores, key=lambda p: p[1], reverse=True)
    if weighted:
        return _weighted_non_maximum_suppression(
            indexed_scores, detections, min_suppression_threshold, min_score)
    else:
        return _non_maximum_suppression(
            indexed_scores, detections, min_suppression_threshold, min_score)


def _overlap_similarity(box1: BBox, box2: BBox) -> float:
    """Return intersection-over-union similarity of two bounding boxes"""
    intersection = box1.intersect(box2)
    if intersection is None:
        return 0.
    intersect_area = intersection.area
    denominator = box1.area + box2.area - intersect_area
    return intersect_area / denominator if denominator > 0. else 0.


def _non_maximum_suppression(
    indexed_scores: List[Tuple[int, float]],
    detections: List[Detection],
    min_suppression_threshold: float,
    min_score: Optional[float]
) -> List[Detection]:
    """Return only most significant detections"""
    kept_boxes: List[BBox] = []
    outputs = []
    for index, score in indexed_scores:
        # exit loop if remaining scores are below threshold
        if min_score is not None and score < min_score:
            break
        detection = detections[index]
        bbox = detection.bbox
        suppressed = False
        for kept in kept_boxes:
            similarity = _overlap_similarity(kept, bbox)
            if similarity > min_suppression_threshold:
                suppressed = True
                break
        if not suppressed:
            outputs.append(detection)
            kept_boxes.append(bbox)
    return outputs


def _weighted_non_maximum_suppression(
    indexed_scores: List[Tuple[int, float]],
    detections: List[Detection],
    min_suppression_threshold: float,
    min_score: Optional[float]
) -> List[Detection]:
    """Return only most significant detections; merge similar detections"""
    remaining_indexed_scores = list(indexed_scores)
    remaining: List[Tuple[int, float]] = []
    candidates: List[Tuple[int, float]] = []
    outputs: List[Detection] = []

    while len(remaining_indexed_scores):
        detection = detections[remaining_indexed_scores[0][0]]
        # exit loop if remaining scores are below threshold
        if min_score is not None and detection.score < min_score:
            break
        num_prev_indexed_scores = len(remaining_indexed_scores)
        detection_bbox = detection.bbox
        remaining.clear()
        candidates.clear()
        weighted_detection = detection
        for (index, score) in remaining_indexed_scores:
            remaining_bbox = detections[index].bbox
            similarity = _overlap_similarity(remaining_bbox, detection_bbox)
            if similarity > min_suppression_threshold:
                candidates.append((index, score))
            else:
                remaining.append((index, score))
        # weighted merging of similar (close) boxes
        if len(candidates):
            weighted = np.zeros((2 + len(detection), 2), dtype=np.float32)
            total_score = 0.
            for index, score in candidates:
                total_score += score
                weighted += detections[index].data * score
            weighted /= total_score
            weighted_detection = Detection(weighted, detection.score)
        outputs.append(weighted_detection)
        # exit the loop if the number of indexed scores didn't change
        if num_prev_indexed_scores == len(remaining):
            break
        remaining_indexed_scores = list(remaining)
    return outputs
