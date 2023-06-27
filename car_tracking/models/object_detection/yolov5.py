from typing import Union

import torch
import numpy as np

from car_tracking.models.object_detection.base import BaseObjectDetector, ObjectDetectorPredictionEntry, BoundingBox


class YOLOv5ObjectDetector(BaseObjectDetector):
    """Class for YOLOv5 object detection model"""

    def __init__(self, weights_path: str, device: Union[str, torch.device]) -> None:
        """
        Initialize YOLOv5 object detector

        Args:
            weights_path: path to the YOLOv5 weights
            device: computing device - either str or torch.device
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, device=device, _verbose=False)

    def __call__(self, img: np.ndarray[np.uint8]) -> list[ObjectDetectorPredictionEntry]:
        """
        Detect people bounding boxes on provided image.

        Args:
            img: np.ndarray[np.uint8] - input image of shape (H, W, C) in RGB colorspace.

        Returns:
            List of ObjectDetectorPredictionEntry instances.
        """
        detections = self.model(img).pred[0]
        result = []
        for det in detections:
            det = det.cpu()
            result.append(ObjectDetectorPredictionEntry(BoundingBox(det[:4], 'xyxy', det[5], det[4])))
        return result
