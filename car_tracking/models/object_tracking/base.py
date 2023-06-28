from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from car_tracking.models.object_detection.base import BoundingBox


@dataclass
class TrackerPredictionEntry:
    bbox: BoundingBox


class BaseTracker(ABC):
    """Base class for all object tracking models implementations"""

    @abstractmethod
    def __call__(
        self, detections: list[BoundingBox], current_frame: np.ndarray[np.uint8]
    ) -> list[TrackerPredictionEntry]:
        """
        Update tracks' statuses based on obtained detections.

        Args:
            detections: list[BoundingBox] - predictions obtained from an object detection model.
            current_frame: np.ndarray[np.uint8] - current frame.

        Returns:
            Bounding box with track ID as TrackingModelPredictionEntry instance for each detected object.
        """

        pass
