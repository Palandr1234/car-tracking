from typing import List, Union, Tuple, Optional

import numpy as np
import mmcv
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from pathlib import Path


BODY_PARTS = ('head', 'torso', 'legs')


def draw_bbox(
    img: np.ndarray,
    bbox: Union[np.ndarray, List[Union[int, float]]],
    color: Union[Tuple[int, int, int], List[int]],
    thickness: int
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
         img: np.ndarray - input image on which to draw bounding box in RGB colorspace.
         bbox: List[Union[int, float]] - coordinates of bounding box in format [xMin, yMin, xMax, yMax].
         color: Union[Tuple[int, int, int], List[int]] - bounding box color. Can be either in (R, G, B) format or
            [R, G, B]. Color scale is [0..255].
         thickness: int - bounding box thickness.

    Returns:
        Copy of input image with drawn bounding box in RGB colorspace as NumPy array .
    """

    img_with_bbox = img.copy()
    x_min, y_min, x_max, y_max = bbox
    img_with_bbox = cv2.rectangle(
        img_with_bbox, (int(x_min), int(y_min)), (int(x_max), int(y_max)), tuple(color), thickness
    )

    return img_with_bbox


def create_cv2_video_writer(video: mmcv.video.io.VideoReader, save_path: str) -> cv2.VideoWriter:
    """
    Create OpenCV video writer.

    Args:
         video: mmcv.video.io.VideoReader - video read with MMCV.
         save_path: str - path where to save output video.

    Returns:
        Configured cv2.VideoWriter object.
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.fps
    size = (video.width, video.height)
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, size)

    return video_writer


def write_text(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_path: Path,
    font_size: Optional[int] = 12,
    background_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    num_padding_pixels: Optional[int] = 2
) -> np.ndarray:
    """
    Write text with background rectangle on given image.

    Args:
        img: np.ndarray - source image in RGB colorspace.
        text: str - text to write on image.
        origin: Tuple[int, int] - text starting point.
        font_path: Path - path to font
        font_size: Optional[int] = 12 - font size.
        background_color: Optional[Tuple[int, int, int]] - background rectangle inner color. Defaults to (0, 0, 0).
        num_padding_pixels: Optional[int] - number of pixels to add on boundaries of background rectangle.
            Defaults to 2.

    Returns:
        Copy if source image with written text.
    """

    color = (255 - background_color[0], 255 - background_color[1], 255 - background_color[2])

    annotated_img = img.copy()
    font = ImageFont.truetype(str(font_path), font_size)
    img_pil = Image.fromarray(annotated_img)
    draw = ImageDraw.Draw(img_pil)
    size_width, size_height = draw.textsize(text, font)
    box_coords = (
        (origin[0], origin[1]),
        (origin[0] + size_width + num_padding_pixels, origin[1] - size_height - num_padding_pixels)
    )
    annotated_img = cv2.rectangle(annotated_img, box_coords[0], box_coords[1], background_color, cv2.FILLED)
    img_pil = Image.fromarray(annotated_img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((origin[0], box_coords[1][1]), text, font=font, fill=color)
    return np.array(img_pil)
