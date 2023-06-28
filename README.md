# Car tracking

## Table of contents
- [Table of contents](#table-of-contents)
- [Task](#task)
- [Data](#Data)
- [Модели](#модели)
- [Обучение](#обучение)
- [Результаты](#результаты)
- [Установка и запуск](#установка-и-запуск)
- [Дальнейшая работа](#дальнейшая-работа)

## Task
Create a CV pipeline for tracking cars

## Data
For creating the dataset, the video that was sent with this task was used. The dataset was labelled manually using CVAT annotation tool. For the annotation, the only label "car" was used. The annotations were downloaded in YOLO format.

### Dataset split
The resulting dataset was randomly split into training, validation and testing datasets. 70% of frames are in the training dataset, 15% - in the validation, 15% - in the testing
