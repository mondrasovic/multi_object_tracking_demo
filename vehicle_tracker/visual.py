#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

from typing import Sequence, Tuple, Dict

import cv2 as cv
import numpy as np

from detection import Detection
from tracking import TrackedDetection

ColorT = Tuple[int, int, int]
PointT = Tuple[int, int]


def labeled_rectangle(
        image: np.ndarray, start_point: PointT, end_point: PointT, label: str,
        rect_color: ColorT, label_color: ColorT, alpha: float = 0.85):
    (x1, y1), (x2, y2) = start_point, end_point

    roi = image[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    image[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)

    font_face = cv.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 2

    (text_width, text_height), baseline = cv.getTextSize(
        label, font_face, font_scale, font_thickness)
    text_rect_end = (
        start_point[0] + text_width, start_point[1] + text_height + baseline)
    cv.rectangle(image, start_point, text_rect_end, rect_color, -1)

    text_start_point = (start_point[0], start_point[1] + text_height + 3)
    cv.putText(
        image, label, text_start_point, font_face, font_scale, label_color,
        font_thickness, cv.LINE_AA)
    cv.rectangle(image, start_point, end_point, rect_color, 2, cv.LINE_AA)


class DetectionVisualizer:
    def __init__(self, class_ids: Sequence[int]) -> None:
        self.colors: Dict[int, ColorT] = {
            class_id: (0, 255, 0) for class_id in class_ids}
    
    def draw_detections(
            self, image: np.ndarray,
            detections: Sequence[Detection]) -> None:
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
            self, image: np.ndarray, tracks: Sequence[TrackedDetection]) -> None:
        for track in tracks:
            text = f'Track ID: {track.track_id}'
            labeled_rectangle(
                image, track.box.top_left, track.box.bottom_right, text,
                (172, 180, 90), (245, 245, 245))
