from abc import ABC, abstractmethod
from pathlib import Path


class BasePipeline(ABC):
    """Base class for all pipelines implementation"""

    @abstractmethod
    def run(self, video_path: Path, save_dir: Path) -> None:
        """
        Run the pipeline on specified video.

        Args:
            video_path: Path - path to the video on which to run the pipeline.
            save_dir: Path - path to the directory where to save results.
        """
        pass
