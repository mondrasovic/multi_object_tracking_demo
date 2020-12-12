#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import json
import dataclasses

from typing import Sequence, Tuple, Dict, Callable, List, Iterable

import cv2 as cv
import numpy as np
from scipy import optimize
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

from bbox import BBox
from detection import Detection


class EmbeddingBuilder:
    INPUT_SIZE = (224, 224)
    
    def __init__(self, config_file_path: str, weights_file_path: str) -> None:
        with open(config_file_path, 'r') as json_file:
            json_content = json.load(json_file)
        
        self._model: keras.models.Model = keras.models.model_from_json(
            json.dumps(json_content))
        self._model.load_weights(weights_file_path)
    
    def build(self, image: np.ndarray, box: BBox) -> np.ndarray:
        roi = self.extract_resized_roi(image, box)
        emb = self._get_embedding(np.array([roi]))[0]
        return emb
    
    def build_batch(
            self,
            image_box_batch: Iterable[Tuple[np.ndarray, BBox]]) -> np.ndarray:
        rois = np.array(
            [self.extract_resized_roi(image, box)
             for image, box in image_box_batch])
        if len(rois) == 0:
            return rois
        emb_batch = self._get_embedding(rois)
        return emb_batch
    
    @staticmethod
    def extract_resized_roi(image: np.ndarray, box: BBox) -> np.ndarray:
        (x1, y1), (x2, y2) = box.top_left, box.bottom_right
    
        x1 = np.clip(x1, 0, image.shape[1] - 1)
        y1 = np.clip(y1, 0, image.shape[0] - 1)
        x2 = np.clip(x2, 0, image.shape[1] - 1)
        y2 = np.clip(y2, 0, image.shape[0] - 1)
    
        roi = image[y1:y2, x1:x2]
        roi = cv.resize(
            roi, EmbeddingBuilder.INPUT_SIZE, interpolation=cv.INTER_LANCZOS4)
        return roi.astype(np.float)
    
    def _get_embedding(self, data_batch: np.ndarray) -> np.ndarray:
        return self._model.predict(preprocess_input(data_batch))


class Track:
    def __init__(self, id_: int, box: BBox, emb: np.ndarray) -> None:
        self._id: int = id_
        self.box: BBox = box
        self.emb: np.ndarray = emb
        self.no_update_count: int = 0
    
    @property
    def id(self) -> int:
        return self._id
    
    def update(self, box: BBox) -> None:
        self.box = box
        self.no_update_count = 0
    
    def notify_no_update(self) -> None:
        self.no_update_count += 1


class TrackBuilder:
    def __init__(self, emb_builder: EmbeddingBuilder) -> None:
        self._emb_builder: EmbeddingBuilder = emb_builder
        self._track_id: int = 1
    
    def build(self, image: np.ndarray, box: BBox) -> Track:
        emb = self._emb_builder.build(image, box)
        track = Track(self._track_id, box, emb)
        self._track_id += 1
        return track


@dataclasses.dataclass(frozen=True)
class TrackedDetection:
    box: BBox
    track_id: int


DetectionsT = Sequence[Detection]
TrackedDetectionsT = List[TrackedDetection]
TracksT = Sequence[Track]
CostEvalT = Callable[[Detection, Track], float]


class TrackingByDetectionMultiTracker:
    def __init__(
            self, emb_config_file_path: str, emb_weights_file_path: str,
            iou_dist_thresh: float = 0.7, emb_dist_thresh: float = 0.2,
            max_no_update_count: int = 30):
        assert 0 <= iou_dist_thresh <= 1
        assert 0 <= emb_dist_thresh <= 1
        assert max_no_update_count > 0
        
        self.iou_dist_thresh: float = iou_dist_thresh
        self.emb_dist_thresh: float = emb_dist_thresh
        self.max_no_update_count: int = max_no_update_count
        
        self._emb_builder: EmbeddingBuilder = EmbeddingBuilder(
            emb_config_file_path, emb_weights_file_path)
        self._track_builder: TrackBuilder = TrackBuilder(self._emb_builder)
        self._tracks: Dict[int, Track] = {}
    
    def track(
            self, image: np.ndarray,
            detections: DetectionsT) -> TrackedDetectionsT:
        tracked_detections = []
        
        # First round. Assign detections according to the IoU distance.
        rem_detections, rem_tracks = self._assign_detections_to_tracks(
            detections, tuple(self._tracks.values()), tracked_detections,
            self._iou_dist, self.iou_dist_thresh)
        
        # Second round. Assign detections according to the distance of
        # visual features (embeddings).
        rem_detections, rem_tracks = self._assign_detections_to_tracks(
            rem_detections, rem_tracks, tracked_detections,
            self._emb_l2_dist(image, detections), self.emb_dist_thresh)
        
        # Mark the remaining tracks as not updated. If no update has been
        # present in a specified number of iterations, then remove the track.
        for track in rem_tracks:
            track.notify_no_update()
            if track.no_update_count >= self.max_no_update_count:
                del self._tracks[track.id]
        
        # The remaining detections will be assigned to new tracks.
        for detection in rem_detections:
            track = self._track_builder.build(image, detection.box)
            self._tracks[track.id] = track
            tracked_detections.append(TrackedDetection(detection.box, track.id))

        return tracked_detections
    
    @staticmethod
    def _assign_detections_to_tracks(
            detections: DetectionsT, tracks: TracksT,
            tracked_detections: TrackedDetectionsT, cost_eval: CostEvalT,
            thresh: float) -> Tuple[DetectionsT, TracksT]:
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
                            
                            track.update(box)
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
    
    def _emb_l2_dist(
            self, image: np.ndarray, detections: DetectionsT) -> CostEvalT:
        emb_batch = self._emb_builder.build_batch(
            ((image, detection.box) for detection in detections))
        emb_cache = dict(zip(detections, emb_batch))
        
        def _cost_eval(detection: Detection, track: Track) -> float:
            return np.linalg.norm(emb_cache[detection] - track.emb)
        
        return _cost_eval
