#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import click

import cv2 as cv
import numpy as np

import skvideo.io

from detection import VehicleDetector
from tracking import TrackingByDetectionMultiTracker
from visual import TrackingVisualizer


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # image = cv.flip(cv.flip(image, 1), 0)
<<<<<<< HEAD
    return cv.resize(image, None, fx=0.8, fy=0.8)
=======
    return cv.resize(image, None, fx=0.6, fy=0.6)
>>>>>>> b224d369b2d4179818f4bf32c250031adc2d92a7


@click.command()
@click.argument('config_file_path')
@click.argument('weights_file_path')
@click.argument('labels_file_path')
@click.option(
    '--input-file-path', '-i', default='',
    help='input file name (if empty, default camera is used)')
@click.option(
    '--score-thresh', '-s', default=0.5, type=float,
    help='detection score (confidence) threshold')
@click.option(
    '--nms-thresh', '-n', default=0.3, type=float,
    help='Non-Maximum Suppression threshold')
@click.option(
    '--use-gpu', '-g', default=False, is_flag=True,
    help='whether to use GPU or not')
def main(
        config_file_path: str, weights_file_path: str, labels_file_path: str,
        input_file_path: str, score_thresh: float, nms_thresh: float,
        use_gpu: bool) -> int:
    detector = VehicleDetector(
        config_file_path, weights_file_path, labels_file_path,
        score_thresh=score_thresh, nms_thresh=nms_thresh, use_gpu=use_gpu)
    tracker = TrackingByDetectionMultiTracker()
    
    tracking_visualizer = TrackingVisualizer()
    
    if input_file_path:
        capture = cv.VideoCapture(input_file_path)
    else:
        capture = cv.VideoCapture(0)
    
    frame_rate = 25
    output_video_file_path = '../video_output.mp4'
    video_writer = skvideo.io.FFmpegWriter(
        output_video_file_path, outputdict={'-r': str(frame_rate)})
    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            break
        
        image = preprocess_image(image)
        detections = detector.detect(image)
        tracks = tracker.track(image, detections)
        tracking_visualizer.draw_tracks(image, tracks)
        
        cv.imshow('Detections preview', image)
        video_writer.writeFrame(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):
            video_writer.close()
            break
    
    capture.release()
    cv.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
