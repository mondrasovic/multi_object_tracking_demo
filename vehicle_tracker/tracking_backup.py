#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses

from typing import Sequence, Tuple, Dict, List

import cv2 as cv
import numpy as np

from scipy import optimize

from bbox import BBox
from detection import Detection


# @dataclasses.dataclass(frozen=True)
# class TrackingResult:
#     box: BBox
#     track_id: int
#
#
# class TrackingByDetectionMultiTracker:
#     def __init__(self):
#         self._trackers: Dict[int, cv.Tracker] = {}
#         self._track_id = 0
#
#     def track(
#             self, image: np.ndarray,
#             detections: Sequence[DetectionResult]) -> Sequence[TrackingResult]:
#         tracking_results = []
#         unassigned_detections = []
#
#         if self._trackers:
#             failed_track_ids = []
#
#             for track_id, tracker in self._trackers.items():
#                 ret, box = tracker.update(image)
#                 if ret:
#                     box = tuple(int(round(c)) for c in box)
#                     tracking_result = TrackingResult(BBox(*box), track_id)
#                     tracking_results.append(tracking_result)
#                 else:
#                     failed_track_ids.append(track_id)
#
#             for track_id in failed_track_ids:
#                 del self._trackers[track_id]
#
#             if not tracking_results or not detections:
#                 return tracking_results
#
#             cost_matrix = []
#
#             for detection in detections:
#                 cost_matrix.append(
#                     [detection.box.intersection_over_union(track.box)
#                      for track in tracking_results])
#
#             cost_matrix = np.array(cost_matrix)
#             row_ind, col_ind = optimize.linear_sum_assignment(-cost_matrix)
#
#             min_iou_thresh = 0.3
#             cost_matrix_pos = 0
#
#             print('*' * 50)
#             print(cost_matrix)
#
#             for row in range(cost_matrix.shape[0]):
#                 if (cost_matrix_pos < len(row_ind) and
#                         row == row_ind[cost_matrix_pos]):
#                     iou = cost_matrix[cost_matrix_pos, cost_matrix_pos]
#                     cost_matrix_pos += 1
#                     if iou >= min_iou_thresh:
#                         continue
#                 unassigned_detections.append(detections[row])
#         else:
#             unassigned_detections = detections
#
#         self._init_trackers(image, unassigned_detections, tracking_results)
#
#         return tracking_results
#
#     def _init_trackers(
#             self, image: np.ndarray, detections: Sequence[DetectionResult],
#             tracking_results: List[TrackingResult]) -> None:
#         for detection in detections:
#             tracker, track_id = self._create_tracker()
#             tracker.init(image, tuple(detection.box))
#             tracking_result = TrackingResult(detection.box, track_id)
#             tracking_results.append(tracking_result)
#
#     def _create_tracker(self) -> Tuple[cv.Tracker, int]:
#         tracker = cv.TrackerCSRT_create()
#         track_id = self._track_id
#         self._trackers[track_id] = tracker
#         self._track_id += 1
#         return tracker, track_id
