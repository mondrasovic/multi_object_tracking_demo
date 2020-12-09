#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses

from typing import Sequence, Tuple, Dict, List, Callable

import cv2 as cv
import numpy as np

from scipy import optimize
from scipy.spatial import distance

from bbox import BBox
from detection import DetectionResult


class ObjectTemplate:
    TEMPLATE_SIZE = (32, 32)
    UPDATE_DECAY = 0.7
    
    def __init__(self, image: np.ndarray, box: BBox) -> None:
        self.template: np.ndarray = self.extract_resized_roi(image, box)
    
    def update(self, image: np.ndarray, box: BBox) -> None:
        roi = self.extract_resized_roi(image, box)
        self.template = (self.template * (1 - self.UPDATE_DECAY) +
                         roi * self.UPDATE_DECAY)
    
    def calc_distance(self, other: 'ObjectTemplate') -> float:
        return distance.cosine(
            self.template.flatten(), other.template.flatten())
    
    @staticmethod
    def extract_resized_roi(image: np.ndarray, box: BBox) -> np.ndarray:
        (x1, y1), (x2, y2) = box.top_left, box.bottom_right
        roi =  image[y1:y2, x1:x2]
        roi = cv.resize(
            roi, ObjectTemplate.TEMPLATE_SIZE, interpolation=cv.INTER_AREA)
        return roi.astype(np.float)


class Track:
    _track_id = 0
    
    def __init__(self, id_: int, image: np.ndarray, box: BBox) -> None:
        self._id: int = id_
        self.box: BBox = box
        self.template = ObjectTemplate(image, box)
    
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


@dataclasses.dataclass(frozen=True)
class TrackingResult:
    box: BBox
    track_id: int


DetectionsT = Sequence[DetectionResult]
TracksT = Sequence[Track]


class TrackingByDetectionMultiTracker:
    def __init__(self):
        self._tracks: Dict[int, Track] = {}
    
    def track(
            self, image: np.ndarray,
            detections: DetectionsT) -> Sequence[TrackingResult]:
        tracking_results = []
        cost_matrix = []
        
        # The abstraction is possible. There are two rounds basically.
        # In the first round, compute the cost matrix using IoU metric. Then,
        # given some threshold, assign detections to tracks. And collect
        # unassigned ones. Once you have another two lists, then run the whole
        # round again but this time use the template distance. This way you
        # could perform all those checks only in one place.
        # TODO Use IoU distance for better interpretability.
        tracks = tuple(self._tracks.values())
        for detection in detections:
            cost_matrix.append(
                [detection.box.intersection_over_union(track.box)
                 for track in tracks])
        
        row_ind, col_ind = [], []
        if detections and self._tracks:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = optimize.linear_sum_assignment(-cost_matrix)
            print('*' * 50)
            print(cost_matrix)
        
        row_ind = list(row_ind)
        col_ind = list(col_ind)
        
        min_iou_thresh = 0.3
        unassigned_detections = []
        
        for i, detection in enumerate(detections):
            if i in row_ind:
                box = detections[i].box
                track_pos = col_ind[row_ind.index(i)]
                track = tracks[track_pos]
                if box.intersection_over_union(track.box) > min_iou_thresh:
                    # track.box = box
                    track.update(image, box)
                    tracking_results.append(TrackingResult(box, track.id))
                    continue
            unassigned_detections.append(detection)

        # unassigned_track_ids = (
        #     track.id for i, track in enumerate(tracks) if i not in col_ind)
        #
        # cost_matrix = []
        # for detection in unassigned_detections:
        #     detection_template = ObjectTemplate(image, detection.box)
        #     for i, track in enumerate(tracks):
        #         distances = []
        #         if i not in col_ind:
        #             distances.append(
        #                 detection_template.calc_distance(track.template))
        
        for detection in unassigned_detections:
            track = self._create_track(image, detection.box)
            tracking_results.append(TrackingResult(detection.box, track.id))
        
        unassigned_track_ids = (
            track.id for i, track in enumerate(tracks) if i not in col_ind)
        for track_id in unassigned_track_ids:
            del self._tracks[track_id]
        
        return tracking_results
    
    def _assign_detections_to_tracks(
            self, detections: DetectionsT,
            tracks: TracksT, cost_eval, thresh) -> Tuple[DetectionsT, TracksT]:
        pass
    
    def _create_track(self, image: np.ndarray, box: BBox) -> Track:
        track = Track.build(image, box)
        self._tracks[track.id] = track
        return track
