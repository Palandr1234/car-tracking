from typing import Optional

import numpy as np
import torchvision
from scipy.optimize import linear_sum_assignment

from car_tracking.models.object_tracking.base import BaseTracker, TrackerPredictionEntry
from car_tracking.models.object_detection import BoundingBox
from car_tracking.models.object_tracking.kalman_box import KalmanBoxTracker


def match_detections_to_tracks(detections: np.ndarray[np.float32], tracks: np.ndarray[np.float32],
                               iou_threshold: float = 0.3) \
        -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8], np.ndarray[np.uint8]]:
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Args:
        detections: list of detecting, bounding boxes in form [x1, y1, x2, y2]
        tracks: current tracks, bounding boxes in form [x1, y1, x2, y2]
        iou_threshold: IoU threshold to match tracks with detections

    Returns:
        np.arrays of matches, unmatched_detections and unmatched_tracks
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=np.uint8), np.arange(len(detections)), np.empty((0, 5), dtype=np.uint8)

    iou_matrix = torchvision.ops.box_iou(detections, tracks).numpy()

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            x, y = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(x, y)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=np.uint8)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class SORTTracker(BaseTracker):
    def __init__(
        self, max_age: Optional[int] = 1, min_hits: Optional[int] = 3, iou_threshold: Optional[float] = 0.3
    ) -> None:
        """
        Initialize SORT tracker.

        Args:
            max_age: Optional[int] - number of frames since the last track's update before track's removal.
                Defaults to 1.
            min_hits: Optional[int] - number of frames during which track remains in initialization phase.
                Defaults to 3.
            iou_threshold: Optional[float] - IoU threshold used for associating detections and tracks. Defaults to 0.3.
        """
        self.max_age = max_age,
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.tracks = []

    def __call__(
        self, detections: list[BoundingBox], current_frame: np.ndarray[np.uint8]
    ) -> list[TrackerPredictionEntry]:
        """
        Update tracks' statuses based on obtained detections.

        Args:
            detections: list[ObjectDetectorPredictionEntry] - predictions obtained from an object detection model.
            current_frame: np.ndarray[np.uint8] - current frame.

        Returns:
            Bounding box with track ID as TrackingModelPredictionEntry instance for each detected object.
        """
        bboxes = detections.copy()
        detections = []
        for bbox in bboxes:
            detections.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.conf_score])
        detections = np.array(detections, dtype=np.float32)
        if not len(detections):
            detections = np.empty((0, 5))

        self.frame_count += 1

        # Get predicted locations from existing tracks.
        tracks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret_bboxes = []
        for i, track in enumerate(tracks):
            pos = self.tracks[i].predict()[0]
            track[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(i)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))
        # Remove tracks with invalid predictions
        for i in reversed(to_del):
            self.tracks.pop(i)
        matched, unmatched_detections, unmatched_tracks = match_detections_to_tracks(detections, tracks,
                                                                                     self.iou_threshold)

        # Update matched tracks with matched detections
        for m in matched:
            self.tracks[m[1]].update(detections[m[0], :])

        # Initialize new tracks for unmatched detections
        for i in unmatched_detections:
            self.tracks.append(KalmanBoxTracker(detections[i, :]))

        for i, track in reversed(self.tracks):
            d = track.get_state()[0]
            if track.time_since_update < 1 and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret_bboxes.append(np.concatenate((d, [track.id + 1])).reshape(1, -1))
            i -= 1

            # Remove dead track
            if track.time_since_update > self.max_age:
                self.tracks.pop(i)

        result = [
            TrackerPredictionEntry(
                bbox=BoundingBox(
                    bbox[0][:4],
                    'xyxy',
                    track_id=np.int32(bbox[0][4]),
                    img_height=current_frame.shape[0],
                    img_width=current_frame.shape[1]
                )
            ) for bbox in ret_bboxes
        ]

        return result
