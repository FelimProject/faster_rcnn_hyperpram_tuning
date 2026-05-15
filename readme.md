# Faster R-CNN — Person Detection (COCO Dataset)

**Algorithm:** Faster R-CNN with ResNet50-FPN backbone  
**Task:** Object Detection — Person class only  
**Dataset:** COCO 2017 (filtered to `person` category)

---

## Overview

This project fine-tunes a pretrained Faster R-CNN (ResNet50 + FPN) model for single-class person detection using the COCO 2017 dataset. The pipeline covers data loading from COCO annotations, preprocessing with Albumentations augmentations, full training and evaluation loops, and hyperparameter tuning via Optuna. The best optimizer configuration is selected based on mAP@50.

---

## Requirements

Install dependencies before running:

```bash
pip install torchmetrics optuna pycocotools
```

Core libraries used:

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | Model, training, and detection utilities |
| `albumentations` | Image augmentation with bounding box support |
| `torchmetrics` | Mean Average Precision (mAP) computation |
| `pycocotools` | COCO annotation parsing |
| `optuna` | Hyperparameter tuning |
| `plotly` | Tuning result visualization |
| `opencv-python` | Bounding box drawing and image display |
| `PIL`, `numpy`, `matplotlib` | Image loading and visualization |

---

## Dataset

- **Source:** COCO 2017 — downloaded via official COCO annotation zip
- **Category:** `person` (category ID = 1) only
- **Split:**
  - Train: 2,000 images (matched against local image directory)
  - Validation: 500 images

### Annotation download

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Expected directory structure

```
dataset/
├── train/
│   └── images/        # COCO 2017 training images
└── val/
    └── images/        # COCO 2017 validation images
annotations/
├── instances_train2017.json
└── instances_val2017.json
```

---

## Project Structure

```
faster_rcnn_computer_vision/
├── faster_rcnn_computer_vision.ipynb   # Main notebook
└── (model checkpoint saved via training loop)
```

---

## Notebook Walkthrough

### 1. Data Loading

- Mounts Google Drive and sets image directory paths
- Verifies GPU availability with `nvidia-smi`
- Imports all required libraries

### 2. Data Cleaning

- Loads COCO train and val annotations using `pycocotools`
- Filters image IDs to those containing at least one `person` annotation
- Matches annotation image IDs against locally available image files to build valid `train_img_ids` and `test_img_ids` lists

### 3. Data Preprocessing

- Sets device to CUDA if available
- Splits to 2,000 training and 500 validation images
- Implements `get_boxes_labels` — converts COCO `[x, y, w, h]` bounding boxes to Pascal VOC `[x1, y1, x2, y2]` format
- Implements `CocoPersonDataset` — custom `Dataset` that loads images from disk and returns image tensors with target dicts (`boxes`, `labels`)
- Defines augmentation pipelines:
  - **Train:** horizontal flip, color jitter, random scale, ImageNet normalization
  - **Val:** ImageNet normalization only
  - Both use `BboxParams(format='pascal_voc', min_visibility=0.3)` to keep bounding boxes consistent with augmented images
- Visualizes random samples with bounding box overlays (before and after augmentation)

### 4. Data Modeling

- Loads pretrained `fasterrcnn_resnet50_fpn` with `DEFAULT` weights
- Unfreezes all backbone parameters for full fine-tuning
- Creates `DataLoader` instances with a custom `collate_fn` (required for variable-size detection targets):
  - Batch size: 4
  - Workers: 2
  - Pin memory: enabled
- Metric: `MeanAveragePrecision` from `torchmetrics`
- Implements:
  - `train_step` — filters out images with no boxes, computes combined detection loss (classifier + bbox regression + RPN), runs backprop, accumulates mAP
  - `eval_step` — computes validation loss in `train` mode (Faster R-CNN requires train mode to return a loss dict), then switches to `eval` mode for mAP computation
  - `train` — full training loop with early stopping (patience = 5), prints metrics every 10 epochs

### 5. Hyperparameter Tuning

- Uses Optuna with `TPESampler` and `MedianPruner`
- Search space:

| Parameter | Range |
|-----------|-------|
| `learning_rate` | log-uniform [1e-6, 1e-2] |
| `optimizer` | Adam, AdamW, SGD |
| `momentum` | [0.85, 0.95] |
| `beta1` | [0.90, 0.99] |
| `beta2` | [0.99, 0.999] |
| `weight_decay` | log-uniform [1e-6, 1e-2] |

- Runs 5 trials, maximizing validation mAP
- Plots optimization history and parameter importances with Plotly
- Best configuration found:

```python
optimizer = SGD(
    model.parameters(),
    lr=0.000254,
    momentum=0.933,
    weight_decay=5.42e-06
)
```

---

## Training Configuration (defaults)

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Early stopping patience | 5 |
| Metric | mAP (COCO-style) |
| Backbone | ResNet50 + FPN (fully unfrozen) |
| Loss | RPN + classification + bbox regression (combined) |
| Device | CUDA if available, else CPU |

---

## Notes

- The notebook is designed for Google Colab with Google Drive mounted at `/content/drive/MyDrive/`.
- Adjust `train_img_dir` and `test_img_dir` in Section 1 if running locally.
- Faster R-CNN returns a loss dict only in `model.train()` mode — `eval_step` explicitly uses `model.train()` with `torch.no_grad()` for loss computation, then switches to `model.eval()` for mAP.
- Images with no person annotations are filtered out during `train_step` to avoid errors from empty target boxes.
- Bounding box augmentation uses `min_visibility=0.3` to discard boxes that become too small after augmentation.
