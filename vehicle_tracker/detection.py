#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import pathlib
import dataclasses

from typing import Sequence, Tuple, Any, Dict, Iterator

import cv2 as cv
import numpy as np


ColorT = Tuple[int, int, int]


class BBox:
    """
    Bounding box with integer coordinates in a 2D image.
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """
        :param x: x coordinate
        :param y: y coordinate
        :param width: a positive width
        :param height: a positive height
        """
        assert (width > 0) and (height > 0)

        self.x, self.y, self.width, self.height = x, y, width, height

        self._center = (self.x + int(round(self.width / 2)),
                        self.y + int(round(self.height / 2)))

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y, self.width, self.height))

    def __eq__(self, other) -> bool:
        return isinstance(other, BBox) and \
               ((self.x == other.x) and (self.y == other.y) and
                (self.width == other.width) and (self.height == other.height))

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.width, self.height))

    def __repr__(self) -> str:
        return f'BBox({self.x},{self.y},{self.width},{self.height})'

    @property
    def top_left(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def top_right(self) -> Tuple[int, int]:
        return self.x + self.width, self.y

    @property
    def bottom_left(self) -> Tuple[int, int]:
        return self.x, self.y + self.height

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self.x + self.width, self.y + self.height
    
    def area(self) -> int:
        return self.width * self.height
    
    def intersection_bbox(self, other: 'BBox') -> 'BBox':
        top_left_x = max(self.x, other.x)
        top_left_y = max(self.y, other.y)
        bottom_right_x = min(self.x + self.width, other.x + other.width)
        bottom_right_y = min(self.y + self.height, other.y + other.height)

        width = bottom_right_x - top_left_x + 1
        height = bottom_right_y - top_left_y + 1

        if min(width, height) <= 0:
            raise ValueError('bounding boxes have no intersection')

        return BBox(top_left_x, top_left_y, width, height)

    def intersection_area(self, other: 'BBox') -> int:
        a = (min(self.x + self.width, other.x + other.width) -
             max(self.x, other.x) + 1)
        b = (min(self.y + self.height, other.y + other.height) -
             max(self.y, other.y) + 1)

        return max(0, a) * max(0, b)

    def intersection_over_union(self, other: 'BBox') -> float:
        intersection_area = self.intersection_area(other)
        union_area = (self.area() + other.area()) - intersection_area
        return intersection_area / float(union_area)


@dataclasses.dataclass(frozen=True)
class DetectionResult:
    boxes: Sequence[BBox]
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


class VehicleDetector(ObjectDetector):
    VALID_LABELS = ('bicycle', 'car', 'motorbike', 'bus', 'train', 'truck')
    
    def __init__(
            self, config_file_path: str, weights_file_path: str,
            labels_file_path: str, *, score_thresh: float = 0.5,
            nms_thresh: float = 0.5, use_gpu: bool = False) -> None:
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
                if score <= self.score_thresh:
                    continue
                    
                box = self.scale_frac_box_to_image_size(
                    detection[0:4], width, height)
                boxes.append(self.box_center_to_top_left(box))
                scores.append(score)
                class_ids.append(class_id)
                class_labels.append(self.labels[class_id])
        
        indices = cv.dnn.NMSBoxes(
            boxes, scores, self.score_thresh, self.nms_thresh).reshape(-1)
        
        boxes = [BBox(*box)
                 for box in self.select_indices_from_nms(boxes, indices)]
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
    def box_center_to_top_left(box: np.ndarray) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        return (
            int(round(x - w / 2.0)), int(round(y - h / 2.0)), int(w), int(h))
    
    @staticmethod
    def select_indices_from_nms(
            arr: Sequence[Any], indices: np.ndarray) -> Sequence[Any]:
        return [arr[i] for i in indices]


class DetectionVisualizer:
    def __init__(self, class_ids: Sequence[int]) -> None:
        self.colors: Dict[int, ColorT] = {
            class_id: (0, 255, 0) for class_id in class_ids}
    
    def draw_detections(
            self, image: np.ndarray, detections: DetectionResult) -> None:
        for bbox, score, class_id, class_label in detections:
            color = tuple(self.colors[class_id])
            top_left = bbox.top_left
            cv.rectangle(
                image, top_left, bbox.bottom_right, color=color,
                thickness=3, lineType=cv.LINE_AA)
            
            text = f'{class_label}: {score:.4f}'
            cv.putText(
                image, text, (top_left[0], top_left[1] - 5),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color,
                thickness=2, lineType=cv.LINE_AA)
