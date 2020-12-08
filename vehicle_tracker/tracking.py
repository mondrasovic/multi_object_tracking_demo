#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses

from typing import Sequence, Tuple

import cv2 as cv
import numpy as np

from bbox import BBox
from detection import DetectionResult


@dataclasses.dataclass(frozen=True)
class TrackingResult:
    box: BBox
    track_id: int


class TrackingByDetectionMultiTracker:
    def __init__(self):
        self._trackers = []
        self._track_id = 0
    
    def track(
            self, image: np.ndarray,
            detections: Sequence[DetectionResult]) -> Sequence[TrackingResult]:
        tracking_results = []
        
        if self._trackers:
            cost_matrix = []
            
            for tracker, track_id in self._trackers:
                ret, box = tracker.update(image)
                box = tuple(int(round(c)) for c in box)
                tracking_result = TrackingResult(BBox(*box), track_id)
                tracking_results.append(tracking_result)
        else:
            for detection in detections:
                tracker_data = self._create_tracker()
                tracker, track_id = tracker_data
                self._trackers.append(tracker_data)
                tracker.init(image, tuple(detection.box))
                tracking_result = TrackingResult(detection.box, track_id)
                tracking_results.append(tracking_result)
        
        return tracking_results
    
    def _create_tracker(self) -> Tuple[cv.Tracker, int]:
        track_id = self._track_id
        self._track_id += 1
        return cv.TrackerCSRT_create(), track_id
