#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import dataclasses

from typing import Sequence

import cv2 as cv
import numpy as np


@dataclasses.dataclass(frozen=True)
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    class_labels: Sequence[str]
    

class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        pass


class VehicleDetector(ObjectDetector):
    def detect(self, frame: np.ndarray) -> DetectionResult:
        pass
