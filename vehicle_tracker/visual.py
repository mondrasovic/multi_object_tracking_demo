#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

from typing import Sequence, Tuple, Dict

import cv2 as cv
import numpy as np

from detection import DetectionResult
from tracking import TrackingResult

ColorT = Tuple[int, int, int]


class DetectionVisualizer:
    def __init__(self, class_ids: Sequence[int]) -> None:
        self.colors: Dict[int, ColorT] = {
            class_id: (0, 255, 0) for class_id in class_ids}
    
    def draw_detections(
            self, image: np.ndarray,
            detections: Sequence[DetectionResult]) -> None:
        for detection in detections:
            color = tuple(self.colors[detection.class_id])
            top_left = detection.box.top_left
            cv.rectangle(
                image, top_left, detection.box.bottom_right, color=color,
                thickness=3, lineType=cv.LINE_AA)
            
            text = f'{detection.class_label}: {detection.score:.4f}'
            cv.putText(
                image, text, (top_left[0], top_left[1] - 5),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color,
                thickness=2, lineType=cv.LINE_AA)


class TrackingVisualizer:
    def draw_tracks(
            self, image: np.ndarray, tracks: Sequence[TrackingResult]) -> None:
        color = (255, 0, 0)
        for track in tracks:
            top_left = track.box.top_left
            
            cv.rectangle(
                image, top_left, track.box.bottom_right, color=color,
                thickness=3, lineType=cv.LINE_AA)
            text = f'track ID: {track.track_id}'
            cv.putText(
                image, text, (top_left[0], top_left[1] - 5),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color,
                thickness=2, lineType=cv.LINE_AA)
