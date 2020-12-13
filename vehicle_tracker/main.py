#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import json
import click
import pathlib

import cv2 as cv
import numpy as np

import skvideo.io

from detection import VehicleDetector
from tracking import TrackingByDetectionMultiTracker
from visual import TrackingVisualizer


def preprocess_image(image: np.ndarray) -> np.ndarray:
    return image
    # return cv.flip(cv.flip(image, 1), 0)
    # return cv.resize(image, None, fx=0.6, fy=0.6)


def get_video_output_path(input_file_path: str) -> str:
    input_path = pathlib.Path(input_file_path)
    output_path = (pathlib.Path('..') /
                   f'{input_path.stem}_processed{input_path.suffix}')
    return str(output_path)


@click.command()
@click.argument('input_file_path')
@click.argument('config_file_path')
def main(input_file_path: str, config_file_path: str) -> int:
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)
    
    detector_config = config['detector']
    min_box_area_ratio = detector_config.get('min_box_area_ratio')
    box_scale = detector_config.get('box_scale')
    detector = VehicleDetector(
        detector_config['config_file_path'],
        detector_config['weights_file_path'],
        detector_config['labels_file_path'],
        score_thresh=detector_config['score_thresh'],
        nms_thresh=detector_config['nms_thresh'],
        min_box_area_ratio=min_box_area_ratio,
        box_scale=box_scale,
        use_gpu=detector_config['use_gpu'])

    tracker_config = config['tracker']
    tracker = TrackingByDetectionMultiTracker(
        tracker_config['config_file_path'],
        tracker_config['weights_file_path'],
        iou_dist_thresh=tracker_config['iou_dist_thresh'],
        emb_dist_thresh=tracker_config['emb_dist_thresh'],
        max_no_update_count=tracker_config['max_no_update_count'])
    
    n_colors = 10
    if 'visualizer' in config:
        visualizer_config = config['visualizer']
        n_colors = visualizer_config.get('colors_number', n_colors)
    tracking_visualizer = TrackingVisualizer(n_colors)
    
    if input_file_path:
        capture = cv.VideoCapture(input_file_path)
    else:
        capture = cv.VideoCapture(0)
    
    frame_rate = 25
    output_video_file_path = get_video_output_path(input_file_path)
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
        
        cv.imshow('Tracking preview', image)
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
