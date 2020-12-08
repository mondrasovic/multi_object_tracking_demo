#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import pathlib
import dataclasses

from typing import Sequence, Tuple, Any

import cv2 as cv
import numpy as np


BoxT = Tuple[int, int, int, int]
ColorT = Tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class DetectionResult:
    boxes: Sequence[BoxT]
    scores: Sequence[float]
    class_ids: Sequence[int]
    class_labels: Sequence[str]
    
    def __iter__(self):
        return iter(
            zip(self.boxes, self.scores, self.class_ids, self.class_labels))


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        pass

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        pass


class VehicleDetector(ObjectDetector):
    VALID_LABELS = ('bicycle', 'car', 'motorbike', 'bus', 'train', 'truck')
    
    def __init__(
            self, config_file_path: str, weights_file_path: str,
            labels_file_path: str, *, confidence: float = 0.5,
            threshold: float = 0.5, use_gpu: bool = False) -> None:
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
        
        self.confidence: float = confidence
        self.threshold: float = threshold

    @property
    def n_classes(self) -> int:
        return len(self.labels)
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        blob = cv.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward(self._layer_names)
        height, width = image.shape[:2]
        return self._extract_prediction_data(outputs, width, height)
    
    def _extract_prediction_data(
            self, outputs, width: int, height: int) -> DetectionResult:
        boxes, scores, class_ids, class_labels = [], [], [], []
        
        for output in outputs:
            for detection in output:
                curr_scores = detection[5:]
                class_id = int(np.argmax(curr_scores))
                if class_id not in self.valid_class_ids:
                    continue
                
                score = float(curr_scores[class_id])
                if score <= self.confidence:
                    continue
                    
                box = self.scale_frac_box_to_image_size(
                    detection[0:4], width, height)
                boxes.append(self.box_center_to_top_left(box))
                scores.append(score)
                class_ids.append(class_id)
                class_labels.append(self.labels[class_id])
        
        indices = cv.dnn.NMSBoxes(
            boxes, scores, self.confidence, self.threshold)
        boxes = self.select_indices_from_nms(boxes, indices)
        scores = self.select_indices_from_nms(scores, indices)
        class_ids = self.select_indices_from_nms(class_ids, indices)
        class_labels = self.select_indices_from_nms(class_labels, indices)
        
        return DetectionResult(boxes, scores, class_ids, class_labels)
    
    @staticmethod
    def scale_frac_box_to_image_size(
            box: np.ndarray, width: int, height: int) -> np.ndarray:
        return np.round(box[0:4] * np.array(
            (width, height, width, height))).astype('int')
    
    @staticmethod
    def box_center_to_top_left(box: np.ndarray) -> BoxT:
        x, y, w, h = box
        return (
            int(round(x - w / 2.0)), int(round(y - h / 2.0)), int(w), int(h))
    
    @staticmethod
    def select_indices_from_nms(
            arr: Sequence[Any], indices: np.ndarray) -> Sequence[Any]:
        return [arr[i[0]] for i in indices]


class DetectionVisualizer:
    def __init__(self, n_classes: int) -> None:
        self.colors: Sequence[ColorT] = [(0, 255, 0)] * n_classes
    
    def draw_detections(
            self, image: np.ndarray, detections: DetectionResult) -> None:
        for box, score, class_id, class_label in detections:
            x, y, w, h = box
            color = tuple(self.colors[class_id])
            cv.rectangle(
                image, (x, y), (x + w, y + h), color=color, thickness=3,
                lineType=cv.LINE_AA)
            
            text = f'{class_label}: {score:.4f}'
            cv.putText(
                image, text, (x, y - 5), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2, lineType=cv.LINE_AA)
