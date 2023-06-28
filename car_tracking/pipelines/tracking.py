from pathlib import Path

import mmcv
import numpy as np
import cv2

from car_tracking.pipelines.base import BasePipeline
from car_tracking.models.object_detection import BaseObjectDetector
from car_tracking.models.object_tracking import BaseTracker, TrackerPredictionEntry
from car_tracking.utils import create_cv2_video_writer, draw_bbox, write_text


class TrackingPipeline(BasePipeline):
    """
    Class that implements the pipeline with object detection and tracking.

    Attributes:
        object_detector: BaseObjectDetector - object detection model.
        tracker: BaseTracker - object tracker.
    """

    def __init__(self, object_detector: BaseObjectDetector, tracker: BaseTracker, font_path: Path) -> None:
        """
        Initialize the tracking pipeline.

        Args:
            object_detector: BaseObjectDetector - object detection model.
            tracker: BaseTracker - object tracker.
            font_path: Path - path to font
        """

        self.object_detector = object_detector
        self.tracker: BaseTracker = tracker
        self.font_path = font_path

        self.tracks_colors: dict[int, tuple[int, int, int]] = {}

    def run(self, video_path: Path, save_dir: Path, **kwargs) -> None:
        """
        Run the tracking pipeline on specified video. Produces the video with bounding boxes and tracks predictions.

        Args:
            video_path: Path - path to the video on which to run the pipeline.
            save_dir: Path - path to the directory where to save results.
        """

        save_dir = save_dir / video_path.stem
        save_dir.mkdir(exist_ok=True)

        video = mmcv.VideoReader(str(video_path))

        # Configure writer of output video
        video_writer = create_cv2_video_writer(video, str(save_dir / 'result.mp4'))

        # Process each frame in input video
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            tracker_predictions = self.detect_and_track(cur_frame)

            # Annotate current frame
            img_rgb = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            img_rgb = self.draw_tracks(img_rgb, tracker_predictions)

            # Write annotated frame to the output video
            video_writer.write(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

        video_writer.release()

    def detect_and_track(self, img: np.ndarray[np.uint8]) -> list[TrackerPredictionEntry]:
        """
        Run object detection and tracking models on given image.

        Args:
            img: np.ndarray - input image.

        Returns:
            Bounding box and track ID for each track.
        """

        object_detector_predictions = self.object_detector(img)
        bboxes = [pred.bbox for pred in object_detector_predictions]
        tracker_predictions = self.tracker(bboxes, img)

        return tracker_predictions

    def draw_tracks(
        self, img: np.ndarray[np.uint8], tracker_predictions: list[TrackerPredictionEntry]
    ) -> np.ndarray[np.uint8]:
        """
        Annotate given image with tracks' bounding boxes and corresponding indices.

        Args:
            img: np.ndarray[np.uint8] - input image.
            tracker_predictions: tracker_predictions: list[TrackerPredictionEntry] - tracker's predictions.

        Returns:
            Annotated copy of input image.
        """

        for pred in tracker_predictions:
            bbox = pred.bbox.xyxy
            track_id = pred.bbox.track_id
            if track_id not in self.tracks_colors:
                r, g, b = np.random.randint(0, 256, 3)
                new_color = (int(r), int(g), int(b))
                self.tracks_colors[track_id] = new_color
            track_color = self.tracks_colors[track_id]
            img = draw_bbox(img, bbox, track_color, 2)
            img = write_text(img, str(track_id), (int(bbox[0] + 5), int(bbox[1] + 5)),
                             self.font_path, background_color=track_color)

        return img
