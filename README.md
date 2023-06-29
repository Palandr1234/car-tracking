# Car tracking

## Table of contents
- [Table of contents](#table-of-contents)
- [Task](#task)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Weights](#weights)
- [Usage](#usage)

## Task
Create a CV pipeline for tracking cars

## Data
For creating the dataset, the video that was sent with this task was used. The dataset was labelled manually using CVAT annotation tool. For the annotation, the only label "car" was used. The annotations were downloaded in YOLO format.

### Dataset split
The resulting dataset was randomly split into training, validation and testing datasets. 70% of frames are in the training dataset, 15% - in the validation, 15% - in the testing

## Models
Two options was tested as object detection model:

1. Custom YOLOv5 trained on the data described above (YOLOv5x was used)
2. Pretrained YOLOv5 

For both models, the implementation was taken from https://github.com/ultralytics/yolov5

For object tracking, SORT tracking was used ([Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763))

## Results
### Metrics
1. Custom model
   | Split | Precision | Recall | mAP50 | mAP50-95 |
   |-------|-----------|--------|-------|----------|
   | Val   | 0.993     | 0.994  | 0.995 | 0.992    |
   | Test  | 0.999     | 0.994  | 0.995 | 0.993    |
2. The metrics for pretrained model was not calculated since it would require relabelling the dataset, and the author did not have the time for it


### Visaulizations
Visualizations for both custom and pretrained models can be found here: https://drive.google.com/drive/folders/1vJDWT1fROW8Uio0nChNWqQFxuaoQekPU?usp=sharing


## Installation
1. Clone the repository
2. Install poetry using https://python-poetry.org/docs/
3. ```
   poetry install
   pip install -U openmim
   mim install mmcv
   ```

## Weights
The weights of the custom model can be downloaded using the following link: https://drive.google.com/file/d/1OHvpYtIHUYVC8VX5VINJFaGIEVXjkNqg/view?usp=sharing

To download the weights of the pretrained yolov5, download any weights from https://github.com/ultralytics/yolov5#pretrained-checkpoints

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