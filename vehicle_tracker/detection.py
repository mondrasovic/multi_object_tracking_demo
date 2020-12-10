#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import pathlib
import dataclasses

from typing import Sequence, Tuple, Any

import cv2 as cv
import numpy as np

from bbox import BBox


@dataclasses.dataclass(frozen=True)
class Detection:
    box: BBox
    score: float
    class_id: int
    class_label: str


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> Detection:
        pass


class VehicleDetector(ObjectDetector):
    VALID_LABELS = ('bicycle', 'car', 'motorbike', 'bus', 'train', 'truck')
    
    def __init__(
            self, config_file_path: str, weights_file_path: str,
            labels_file_path: str, *, score_thresh: float = 0.5,
            nms_thresh: float = 0.3, use_gpu: bool = False) -> None:
        self.labels: Sequence[str] = pathlib.Path(
            labels_file_path).read_text().strip().split()
        self.valid_class_ids = set(
            i
            for i, label in enumerate(self.labels)
            if label in self.VALID_LABELS)
        
        self._net = cv.dnn.readNetFromDarknet(
            config_file_path, weights_file_path)
    
        self._layer_names: Sequence[str] = self._net.getLayerNames()
        self._layer_names = [
            self._layer_names[i[0] - 1]
            for i in self._net.getUnconnectedOutLayers()]
    
        if use_gpu:
            self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        
        self.score_thresh: float = score_thresh
        self.nms_thresh: float = nms_thresh
    
    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        blob = cv.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward(self._layer_names)
        height, width = image.shape[:2]
        return self._extract_prediction_data(outputs, width, height)
    
    def _extract_prediction_data(
            self, outputs, width: int,
            height: int) -> Sequence[Detection]:
        boxes, scores, class_ids, class_labels = [], [], [], []
        
        for output in outputs:
            for detection in output:
                curr_scores = detection[5:]
                class_id = int(np.argmax(curr_scores))
                if class_id not in self.valid_class_ids:
                    continue
                
                score = float(curr_scores[class_id])
                if score <= self.score_thresh:
                    continue

                # TODO Make this an optional parameter.
                box = self.scale_frac_box_to_image_size(
                    detection[0:4], width, height)
                area_ratio = (box[2] * box[3]) / float(width * height)
                if area_ratio < 0.002:
                    continue
                
                boxes.append(self.box_center_to_top_left(box))
                scores.append(score)
                class_ids.append(class_id)
                class_labels.append(self.labels[class_id])
        
        if not boxes:
            return []
        
        indices = cv.dnn.NMSBoxes(
            boxes, scores, self.score_thresh, self.nms_thresh).reshape(-1)
        
        boxes = self.select_indices_from_nms(boxes, indices)
        scores = self.select_indices_from_nms(scores, indices)
        class_ids = self.select_indices_from_nms(class_ids, indices)
        class_labels = self.select_indices_from_nms(class_labels, indices)
        
        detection_results = [
            Detection(BBox(*box), score, class_id, class_label)
            for box, score, class_id, class_label in
            zip(boxes, scores, class_ids, class_labels)]
        
        return detection_results
    
    @staticmethod
    def scale_frac_box_to_image_size(
            box: np.ndarray, width: int, height: int) -> np.ndarray:
        return np.round(box[0:4] * np.array(
            (width, height, width, height))).astype('int')
    
    @staticmethod
    def box_center_to_top_left(box: np.ndarray) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        return (
            int(round(x - w / 2.0)), int(round(y - h / 2.0)), int(w), int(h))
    
    @staticmethod
    def select_indices_from_nms(
            arr: Sequence[Any], indices: np.ndarray) -> Sequence[Any]:
        return [arr[i] for i in indices]
