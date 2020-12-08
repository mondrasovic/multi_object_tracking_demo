#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import dataclasses

from typing import Sequence

import numpy as np
import cv2 as cv

from bbox import BBox
from detection import DetectionResult


@dataclasses.dataclass(frozen=True)
class TrackingResult:
    detection_result: DetectionResult
    track_id: int


class SingleObjectTracker(abc.ABC):
    pass


class MultiObjectTracker(abc.ABC):
    @abc.abstractmethod
    def track(self, image: np.ndarray, **args) -> Sequence[TrackingResult]:
        pass


class TrackingByDetectionMultiTracker(MultiObjectTracker):
    def track(self, image: np.ndarray, **kwargs) -> Sequence[TrackingResult]:
        pass
