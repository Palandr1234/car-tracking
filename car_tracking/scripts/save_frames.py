import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    """
    Parse arguments (video path, saving directory and frame step)

    Returns:
        argparse.Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to tthe video",
    )
    parser.add_argument(
        "save_dir",
        type=Path,
        help="Path to the directory to save the frames",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Frame step with which frames should be saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video_path))
    frame_num = 0
    while True:
        success, frame = capture.read()
        if success and frame_num % args.frame_step == 0:
            cv2.imwrite(str(args.save_dir) + "/frame_" + "0" * (6 - len(str(frame_num))) + f"{frame_num}.PNG", frame)
        elif not success:
            break
        frame_num = frame_num + 1
    capture.release()
