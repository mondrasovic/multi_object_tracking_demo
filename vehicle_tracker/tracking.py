#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses

from typing import Sequence, Tuple, Dict, Callable, List

import cv2 as cv
import numpy as np
from scipy import optimize
from scipy.spatial import distance

from bbox import BBox
from detection import Detection


# class VelocityModel:
#     def __init__(self, position: np.ndarray, friction: float = 0.7) -> None:
#         assert 0 <= friction <= 1
#
#         self.position: np.ndarray = position.copy()
#         self.velocity: np.ndarray = np.array([0, 0])
#         self.friction: float = friction
#
#     def update(self, position: np.ndarray) -> None:
#         self.velocity = (self.friction * self.velocity +
#                          (1 - self.friction) * (position - self.position))
#         self.position = position


# class ObjectTemplate:
#     TEMPLATE_SIZE = (64, 64)
#     UPDATE_DECAY = 0.7
#
#     def __init__(self, image: np.ndarray, box: BBox) -> None:
#         self.template: np.ndarray = self.extract_resized_roi(image, box)
#
#     def update(self, image: np.ndarray, box: BBox) -> None:
#         roi = self.extract_resized_roi(image, box)
#         self.template = (self.template * (1 - self.UPDATE_DECAY) +
#                          roi * self.UPDATE_DECAY)
#
#     def calc_dist(self, other: 'ObjectTemplate') -> float:
#         # return 1 - ssim(self.template, other.template, multichannel=True)
#         return distance.cosine(
#             self.template.flatten(), other.template.flatten())
#
#     @staticmethod
#     def extract_resized_roi(image: np.ndarray, box: BBox) -> np.ndarray:
#         (x1, y1), (x2, y2) = box.top_left, box.bottom_right
#
#         x1 = np.clip(x1, 0, image.shape[1] - 1)
#         y1 = np.clip(y1, 0, image.shape[0] - 1)
#         x2 = np.clip(x2, 0, image.shape[1] - 1)
#         y2 = np.clip(y2, 0, image.shape[0] - 1)
#
#         roi = image[y1:y2, x1:x2]
#         roi = cv.resize(
#             roi, ObjectTemplate.TEMPLATE_SIZE, interpolation=cv.INTER_AREA)
#         return roi.astype(np.float)


class ObjectTemplate:
    TEMPLATE_SIZE = (64, 64)
    UPDATE_DECAY = 0.7
    
    def __init__(self, image: np.ndarray, box: BBox) -> None:
        self.template: np.ndarray = self.extract_resized_roi(image, box)
    
    def update(self, image: np.ndarray, box: BBox) -> None:
        roi = self.extract_resized_roi(image, box)
        self.template = (self.template * (1 - self.UPDATE_DECAY) +
                         roi * self.UPDATE_DECAY)
    
    def calc_dist(self, other: 'ObjectTemplate') -> float:
        # return 1 - ssim(self.template, other.template, multichannel=True)
        return distance.cosine(
            self.template.flatten(), other.template.flatten())
    
    @staticmethod
    def extract_resized_roi(image: np.ndarray, box: BBox) -> np.ndarray:
        (x1, y1), (x2, y2) = box.top_left, box.bottom_right
        
        x1 = np.clip(x1, 0, image.shape[1] - 1)
        y1 = np.clip(y1, 0, image.shape[0] - 1)
        x2 = np.clip(x2, 0, image.shape[1] - 1)
        y2 = np.clip(y2, 0, image.shape[0] - 1)
        
        roi = image[y1:y2, x1:x2]
        roi = cv.resize(
            roi, ObjectTemplate.TEMPLATE_SIZE, interpolation=cv.INTER_AREA)
        return roi.astype(np.float)


class Track:
    _track_id = 1
    
    def __init__(self, id_: int, image: np.ndarray, box: BBox) -> None:
        self._id: int = id_
        self.box: BBox = box
        self.template = ObjectTemplate(image, box)
        self.velocity = VelocityModel(np.array(box.center))
        self.no_update_count = 0
    
    @staticmethod
    def build(image: np.ndarray, box: BBox) -> 'Track':
        track = Track(Track._track_id, image, box)
        Track._track_id += 1
        return track
    
    @property
    def id(self):
        return self._id
    
    def update(self, image: np.ndarray, box: BBox) -> None:
        self.box = box
        self.template.update(image, box)
        self.velocity.update(np.array(box.center))
        self.no_update_count = 0
    
    def notify_no_update(self) -> None:
        self.no_update_count += 1


@dataclasses.dataclass(frozen=True)
class TrackedDetection:
    box: BBox
    track_id: int


DetectionsT = Sequence[Detection]
TrackedDetectionsT = List[TrackedDetection]
TracksT = Sequence[Track]
CostEvalT = Callable[[Detection, Track], float]


def draw_velocity(image: np.ndarray, track: Track) -> None:
    velocity = track.velocity.velocity
    cv.putText(
        image, f'[{velocity[0]:.4f}, {velocity[1]:.4f}]', track.box.center,
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)


class TrackingByDetectionMultiTracker:
    def __init__(
            self, iou_dist_thresh: float = 0.7,
            template_dist_thresh: float = 0.15, max_no_update_count: int = 30):
        assert 0 <= iou_dist_thresh <= 1
        assert 0 <= template_dist_thresh <= 1
        assert max_no_update_count > 0
        
        self.iou_dist_thresh: float = iou_dist_thresh
        self.template_dist_thresh: float = template_dist_thresh
        self.max_no_update_count: int = max_no_update_count
        
        self._tracks: Dict[int, Track] = {}
    
    def track(
            self, image: np.ndarray,
            detections: DetectionsT) -> TrackedDetectionsT:
        tracked_detections = []
        
        # First round. Assign detections according to the IoU distance.
        rem_detections, rem_tracks = self._assign_detections_to_tracks(
            image, detections, tuple(self._tracks.values()), tracked_detections,
            self._iou_dist, self.iou_dist_thresh)
        
        # Second round. Assign detections according to the distance of
        # visual features.
        rem_detections, rem_tracks = self._assign_detections_to_tracks(
            image, rem_detections, rem_tracks, tracked_detections,
            self._template_cosine_dist(image, detections),
            self.template_dist_thresh)
        
        # Mark the remaining tracks as not updated. If no update has been
        # present in a specified number of iterations, then remove the track.
        for track in rem_tracks:
            track.notify_no_update()
            if track.no_update_count >= self.max_no_update_count:
                del self._tracks[track.id]
        
        # The remaining detections will be assigned to new tracks.
        for detection in rem_detections:
            track = self._create_track(image, detection.box)
            tracked_detections.append(TrackedDetection(detection.box, track.id))

        return tracked_detections
    
    @staticmethod
    def _assign_detections_to_tracks(
            image: np.ndarray, detections: DetectionsT,
            tracks: TracksT, tracked_detections: TrackedDetectionsT,
            cost_eval: CostEvalT, thresh: float) -> Tuple[DetectionsT, TracksT]:
        rem_detections = []
        assigned_tracks_pos = set()
        
        if detections:
            cost_matrix = []
            
            for detection in detections:
                cost_matrix.append(
                    [cost_eval(detection, track) for track in tracks])
            
            row_ind = col_ind = []
            if tracks:
                cost_matrix = np.array(cost_matrix)
                row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
            
            assignment_pos = 0
            for cost_matrix_row, detection in enumerate(detections):
                if assignment_pos < len(row_ind):
                    assigned_row = row_ind[assignment_pos]
                    
                    if cost_matrix_row == assigned_row:
                        assigned_col = col_ind[assignment_pos]
                        cost = cost_matrix[assigned_row, assigned_col]
                        assignment_pos += 1
                        if cost < thresh:
                            box = detections[assigned_row].box
                            track = tracks[assigned_col]
                            
                            track.update(image, box)
                            assigned_tracks_pos.add(assigned_col)
                            tracked_detections.append(
                                TrackedDetection(box, track.id))
                            
                            continue
                rem_detections.append(detection)
        
        rem_tracks_pos = set(range(len(tracks))) - assigned_tracks_pos
        rem_tracks = [tracks[i] for i in rem_tracks_pos]
        
        return rem_detections, rem_tracks
    
    @staticmethod
    def _iou_dist(detection: Detection, track: Track) -> float:
        return 1 - detection.box.intersection_over_union(track.box)
    
    @staticmethod
    def _template_cosine_dist(
            image: np.ndarray, detections: DetectionsT) -> CostEvalT:
        templates_cache = {
            detection: ObjectTemplate(image, detection.box)
            for detection in detections}
        
        def _cost_eval(detection: Detection, track: Track) -> float:
            return templates_cache[detection].calc_dist(track.template)
        
        return _cost_eval
    
    def _create_track(self, image: np.ndarray, box: BBox) -> Track:
        track = Track.build(image, box)
        self._tracks[track.id] = track
        return track
