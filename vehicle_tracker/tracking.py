#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses

from typing import Sequence, Tuple, Dict

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
        self._trackers: Dict[int, cv.Tracker] = {}
        self._track_id = 0
    
    def track(
            self, image: np.ndarray,
            detections: Sequence[DetectionResult]) -> Sequence[TrackingResult]:
        tracking_results = []
        
        if self._trackers:
            failed_track_ids = []
            for track_id, tracker in self._trackers.items():
                ret, box = tracker.update(image)
                if not ret:
                    failed_track_ids.append(track_id)
                else:
                    box = tuple(int(round(c)) for c in box)
                    tracking_result = TrackingResult(BBox(*box), track_id)
                    tracking_results.append(tracking_result)
            
            for track_id in failed_track_ids:
                del self._trackers[track_id]
            
            if tracking_results:
                cost_matrix = []
                
                for detection in detections:
                    cost_matrix.append(
                        [1 - detection.box.intersection_over_union(track.box)
                         for track in tracking_results])
            
                cost_matrix = np.array(cost_matrix)
        else:
            for detection in detections:
                tracker, track_id = self._create_tracker()
                self._trackers[track_id] = tracker
                tracker.init(image, tuple(detection.box))
                tracking_result = TrackingResult(detection.box, track_id)
                tracking_results.append(tracking_result)
        
        return tracking_results
    
    def _create_tracker(self) -> Tuple[cv.Tracker, int]:
        track_id = self._track_id
        self._track_id += 1
        return cv.TrackerCSRT_create(), track_id
