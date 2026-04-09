# Object Detection — Inference & Explainability

Pipeline de inferencia y explicabilidad para modelos **Faster R-CNN** preentrenados, con soporte para múltiples backbones y métodos XAI.

## Features

- **Backbones:** ResNet-50, Swin Transformer, custom VGG16
- **LoRA** support
- **XAI:** GradCAM (HiResCAM), LIME, SHAP
- **Configuration:** YAML files via OmegaConf
- **Data format:** COCO JSON

## Directory Structure

```
.
├── config/
│   └── config_faster.yaml      # Configuration for Faster R-CNN
├── data/
│   ├── train/
│   ├── test/
│   ├── validation/
│   └── labels.json             # COCO-format annotations
├── models/
│   └── fasterrcnn/             # Saved model checkpoints (.ckpt)
├── output/                     # Inference and XAI output images
├── src/
│   └── faster_rcnn/
│       ├── inference.py
│       ├── pt_lightning/
│       │   ├── classes.py      # LightningModule and Dataset
│       │   └── utils.py        # Model builder, transforms, image saving
│       └── swin_utils/
│           └── build.py        # Swin Transformer backbone
├── inference.py                # Run inference on test images
├── xai.py                      # Explainability pipeline
└── requirements.txt
```

## Dataset Format

Data must be organized under `data/` with a single `labels.json` in COCO format:

```
data/
├── test/
└── labels.json
```

```json
{
    "images": [
        {"id": 1, "file_name": "image.png", "width": 652, "height": 1072}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 0, "bbox": [186, 468, 188, 77], "area": 14476, "iscrowd": 0}
    ],
    "categories": [
        {"id": 0, "name": "object"}
    ]
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config_faster.yaml` before running any script:

```yaml
size: 800
inf_name: "fasterrcnn_lora-True_bs-4_lr-0.0003_size-800-val_map=0.90.ckpt"

model:
  nclasses: 2
  model_type: "fasterrcnn"   # fasterrcnn | swin | custom
  backbone_name: "swin_base_patch4_window7_224"  # only used for swin
  lora: true

paths:
  test: "data/test/"
  labels: "data/labels.json"
  output_images: "output/"
  model_path: "models/fasterrcnn"
```

The `inf_name` field encodes the model configuration — the code parses it to reconstruct the architecture automatically.

## Usage

> This project does **not** include training scripts. It expects a pretrained `.ckpt` checkpoint placed in the path specified by `model_path` in the config.

### Inference

Runs detection on a random test image and saves the result with bounding boxes to `output/output.png`.

```bash
python inference.py
```

Red boxes = predictions, green boxes = ground truth.

### Explainability (XAI)

```bash
# GradCAM on a random image (default)
python xai.py

# Specific method and image index
python xai.py --method gradcam --mode single --index 5

# All methods on the full test set
python xai.py --method all --mode dataset

# Force CPU (e.g. when CUDA is unavailable)
python xai.py --method gradcam --device cpu
```

| Argument | Options | Default | Description |
|---|---|---|---|
| `--method` | `gradcam`, `lime`, `shap`, `all` | `gradcam` | Explainability method |
| `--mode` | `single`, `dataset` | `single` | Single image or full test set |
| `--index` | integer | random | Image index (single mode only) |
| `--device` | `cuda`, `cpu` | auto | Device selection (auto-detects CUDA) |

Output is saved under `output/<image_index>/`.

**GradCAM** generates one heatmap per target layer combination (14 total).  
**LIME** highlights superpixel regions most relevant to the detection score.  
**SHAP** produces a heatmap of feature attributions aggregated across channels.

## Model Backbones

| `model_type` | Backbone | Notes |
|---|---|---|
| `fasterrcnn` | ResNet-50 + FPN | Default, pretrained on ImageNet |
| `swin` | Swin Transformer | Requires `backbone_name` in config |
| `custom` | VGG16 | Single-scale RoI pooling |
