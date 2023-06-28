from pathlib import Path

import hydra
from omegaconf import DictConfig

from car_tracking.pipelines.base import BasePipeline


@hydra.main(version_base=None, config_path='../../configs', config_name='tracking.yaml')
def main(config: DictConfig) -> None:
    save_dir = Path(config.data.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pipeline: BasePipeline = hydra.utils.instantiate(config.pipeline)
    pipeline.run(Path(config.data.path), save_dir)


if __name__ == '__main__':
    main()
