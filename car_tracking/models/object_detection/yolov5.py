from typing import Union, Optional

import torch
import numpy as np

from car_tracking.models.object_detection.base import BaseObjectDetector, ObjectDetectorPredictionEntry, BoundingBox


class YOLOv5ObjectDetector(BaseObjectDetector):
    """Class for YOLOv5 object detection model"""

    def __init__(self, weights_path: str, device: Union[str, torch.device], conf_thresh: float,
                 target_classes: Optional[list[int]] = None) -> None:
        """
        Initialize YOLOv5 object detector

        Args:
            weights_path: path to the YOLOv5 weights
            device: computing device - either str or torch.device
            conf_thresh: confidence threshold
            target_classes: target classes of the model; if None, all classes are considered
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, device=device, _verbose=False)
        self.model.conf = conf_thresh
        self.target_classes = target_classes

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
            if self.target_classes is None or det[-1] in self.target_classes:
                result.append(ObjectDetectorPredictionEntry(BoundingBox(det[:4], 'xyxy', det[5], det[4])))
        return result
