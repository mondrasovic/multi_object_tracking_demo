#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import click

import cv2 as cv

from detection import VehicleDetector, DetectionVisualizer


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
    '--nms-thresh', '-s', default=0.5, type=float,
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
    detection_visualizer = DetectionVisualizer(detector.valid_class_ids)
    
    if input_file_path:
        capture = cv.VideoCapture(input_file_path)
    else:
        capture = cv.VideoCapture(0)
    
    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            break
        
        detections = detector.detect(image)
        detection_visualizer.draw_detections(image, detections)
        cv.imshow('Detections preview', image)
        key = cv.waitKey(10) & 0xff
        if key == ord('q'):
            break
    
    capture.release()
    cv.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
