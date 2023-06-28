# Car tracking

## Table of contents
- [Table of contents](#table-of-contents)
- [Task](#task)
- [Data](#Data)
- [Модели](#модели)
- [Обучение](#обучение)
- [Результаты](#результаты)
- [Installation](#installation)
- [Usage](#usage)
- [Дальнейшая работа](#дальнейшая-работа)

## Task
Create a CV pipeline for tracking cars

## Data
For creating the dataset, the video that was sent with this task was used. The dataset was labelled manually using CVAT annotation tool. For the annotation, the only label "car" was used. The annotations were downloaded in YOLO format.

### Dataset split
The resulting dataset was randomly split into training, validation and testing datasets. 70% of frames are in the training dataset, 15% - in the validation, 15% - in the testing

## Installation
1. Clone the repository
2. ```
   poetry install
   pip install -U openmim
   mim install mmcv
   ```

## Usage
In ```configs/tracking.yaml``` specify the following lines:
* ```device``` - computing device, ```cuda:{0, 1, 2}``` or ```cpu```
* ```pipeline.object_detector.weights_path``` - weights to object detector weights
* ```pipeline.object_detector.target_classes``` - ```[0]``` if using custom model and ```[1, 2, 3, 5, 7]``` if using pretrained yolov5
* ```data.path``` - video path
* ```data_save_dir``` - saving directory

To use it as script:

```python -m car_tracking.scripts.run_pipeline```

To use it in the code:
```python
from pathlib import Path
import hydra

from car_tracking.pipelines.base import BasePipeline


with hydra.initialize(version_base=None, config_path='/path/to/configs'):
     config = hydra.compose(config_name='tracking')
pipeline: BasePipeline = hydra.utils.instantiate(config.model)
video_path = Path("/path/to/video")
save_dir = Path("/path/to/dir")
save_dir.mkdir(exist_ok=True, parents=True)
pipeline.run(video_path, save_dir)
```