from dataclasses import dataclass
from typing import Union, Optional
from abc import ABC, abstractmethod

import numpy as np


class BoundingBox:
    """
    Class implementing the logic of bounding box.

    Attributes:
        x1: `np.float32` - X-coordinate of the top-left bounding box' point.
        y1: `np.float32` - Y-coordinate of the top-left bounding box' point.
        x2: `np.float32` - X-coordinate of the bottom-right bounding box' point.
        y2: `np.float32` - Y-coordinate of the bottom-right bounding box' point.
        w: `np.float32` - bounding box' width.
        h: `np.float32` - bounding box' height.
        conf_score: `np.float32` - confidence score of bounding box. Can be `None` if not provided.
        track_id: np.int32 - track ID.
    """

    SUPPORTED_MODES = ('xyxy', 'xywh')

    def __init__(
        self,
        coords: Union[list[Union[float, int]], np.ndarray[Union[np.float32, np.int32]]],
        coords_mode: Optional[str] = 'xyxy',
        class_id: Optional[Union[int, np.int32]] = None,
        conf_score: Optional[Union[float, np.float32]] = None,
        track_id: Optional[Union[int, np.int32]] = None,
        img_height: Optional[Union[int, np.int32]] = None,
        img_width: Optional[Union[int, np.int32]] = None
    ) -> None:
        """
        Initialize the bounding box.

        Args:
            coords: Union[list[Union[float, int]], np.ndarray[Union[np.float32, np.int32]]] - array / list of four
                bounding box coordinates.
            coords_mode: str - bounding box' coordinates mode. Can be either 'xyxy' or 'xywh'. Defaults to 'xywh'.
            class_id: Union[int, np.int32] - class ID of an object within this bounding box. Defaults to None.
            conf_score: Optional[Union[float, np.float32]] - bounding box' confidence score. Defaults to None.
            track_id: Optional[Union[int, np.int32]] - track ID. Defaults to None.
            img_height: Optional[Union[int, np.int32]] - source image's height. Defaults to None.
            img_width: Optional[Union[int, np.int32]] - source image's width. Defaults to None.
        """

        assert len(coords) == 4, "Bounding box should have four coordinates."
        assert coords_mode in self.SUPPORTED_MODES, f"Bounding box should have mode one of {self.SUPPORTED_MODES}, " \
                                                    f"but got: {coords_mode}"
        coords = np.array(coords, dtype=np.float32)
        self.x1, self.y1 = coords[:2]
        self.x2, self.y2 = coords[2:] if coords_mode == 'xyxy' else coords[:2] + coords[2:]
        if img_height and img_width:
            self.x1 = np.clip(self.x1, 0, img_width)
            self.y1 = np.clip(self.y1, 0, img_height)
            self.x2 = np.clip(self.x2, 0, img_width)
            self.y2 = np.clip(self.y2, 0, img_height)
        self.w, self.h = self.x2 - self.x1, self.y2 - self.y1
        self.class_id = np.int32(class_id) if class_id is not None else class_id
        self.conf_score = np.float32(conf_score) if conf_score is not None else conf_score
        self.track_id = np.int32(track_id) if track_id is not None else track_id

    @property
    def xyxy(self) -> np.ndarray[np.float32]:
        """
        Get the XYXY bounding box' representation

        Returns:
            NumPy array of bounding box' coordinates with `np.float32` type in XYXY format.
        """

        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray[np.float32]:
        """
        Get the XYWH bounding box' representation

        Returns:
            NumPy array of bounding box' coordinates with `np.float32` type in XYWH format.
        """

        return np.array([self.x1, self.y1, self.w, self.h], dtype=np.float32)

    @property
    def area(self) -> np.float32:
        """
        Get the bounding box' area.

        Returns:
            Bounding box' area with `np.float32` type.
        """

        return self.w * self.h


@dataclass
class ObjectDetectorPredictionEntry:
    bbox: BoundingBox


class BaseObjectDetector(ABC):
    """Base class for all object detection models implementations"""

    @abstractmethod
    def __call__(self, img: np.ndarray[np.uint8]) -> list[ObjectDetectorPredictionEntry]:
        """
        Detect people bounding boxes on provided image.

        Args:
            img: np.ndarray[np.uint8] - input image of shape (H, W, C) in RGB colorspace.

        Returns:
            List of ObjectDetectorPredictionEntry instances.
        """
        pass
