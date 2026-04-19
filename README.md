# Tulip Object Detection Project IT3B

This repository contains the shared codebase and assets for the IT3B tulip object detection project. It brings the group work together in one place while keeping each model in its own folder for comparison.

The repository is intended to support the deliverable by providing:
- the project code
- the prepared dataset used for experiments
- model-specific training and evaluation scripts
- saved model weights
- a shared structure for comparing different detection approaches

The written report is intentionally not included here and is submitted separately.

## Project Scope

The project focuses on tulip detection in agricultural field imagery. The aim is to compare object detection approaches on the same tulip dataset while keeping the implementation organized and reproducible.

At the time of submission, the repository includes:
- a Faster R-CNN workflow
- a RetinaNet workflow
- placeholder folders for two YOLO-based student contributions
- shared utilities for dataset loading, metrics, configuration, and visualization
- preprocessing utilities
- trained `.pth` weight files for Faster R-CNN and RetinaNet

## Repository Structure

```text
Tulip_object_detection_project_IT3B/
├── src/
│   ├── Datasets/
│   │   ├── Gold/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── Silver/
│   ├── models/
│   │   ├── faster_rcnn/
│   │   ├── retinanet/
│   │   ├── yolo_student1/
│   │   └── yolo_student2/
│   ├── common/
│   ├── preprocessing/
│   └── main.py
├── weights/
├── results/
├── requirements.txt
└── README.md
```

## Included Dataset

The annotated dataset used in this repository is stored under:

- `src/Datasets/Gold/train/`
- `src/Datasets/Gold/val/`
- `src/Datasets/Gold/test/`

Each split contains:
- an `images/` folder
- a `_annotations.coco.json` COCO annotation file

This means the repository already includes the local annotated dataset assets needed for the project package.

## Included Models

### Faster R-CNN

Location:
- `src/models/faster_rcnn/`

Files:
- `train_faster_rcnn.py`
- `evaluate_faster_rcnn.py`
- `predict_faster_rcnn.py`
- `utils.py`

Saved weight:
- `weights/faster_rcnn/fasterrcnn_tulip_improved.pth`

### RetinaNet

Location:
- `src/models/retinanet/`

Files:
- `train_retinanet.py`
- `evaluate_retinanet.py`
- `predict_retinanet.py`
- `utils.py`

Saved weight:
- `weights/retinanet/retinanet_tulip.pth`

### YOLO Student Folders

Locations:
- `src/models/yolo_student1/`
- `src/models/yolo_student2/`

These folders are included so the final repository structure supports all group models in one place. At this stage they serve as integration folders for the group submission.

## Shared Utilities

The shared code under `src/common/` provides:
- dataset handling
- configuration
- metrics
- visualization helpers

The preprocessing folder contains:
- `data_splitting.py`
- `image_sampler.py`

## Results and Weights

The repository includes dedicated folders for:

- `weights/`
  This stores trained model files.
- `results/tables/`
  Intended for metric tables and summary outputs.
- `results/figures/`
  Intended for plots and qualitative prediction images.
- `results/logs/`
  Intended for training and evaluation logs.

## Setup

Use Python 3.10 or a compatible environment.

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you are using the same local environment as during development, activate the environment first and then run the commands from the repository root.

## Example Commands

From the repository root:

```powershell
python src\models\faster_rcnn\train_faster_rcnn.py
python src\models\retinanet\train_retinanet.py
```

## Deliverable Alignment

This repository is structured to cover the code-side parts of the deliverable:

- project repository with organized folder structure
- source code for model development
- annotated dataset included in the repository structure
- separate folders for different model approaches
- saved trained model weights
- utilities for preprocessing, metrics, and visualization
- space for results, logs, and figures

Items handled outside this repository:
- the written report
- final written analysis and conclusions document

## Notes for Reviewers

- The repository combines the main experimental models into one shared submission structure.
- Faster R-CNN and RetinaNet contain the actual working experiment scripts used during development.
- The report should be read together with this repository, since the report explains the experimental findings and interpretation.

## Submission Note

For submission, this folder is the repository package to upload to GitHub:

`D:\Important things\Important things\Studies\Inholland\Year 3\Data and AI\Term 1\New\Project\Tulip_object_detection_project_IT3B`
