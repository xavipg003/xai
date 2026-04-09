from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from src.faster_rcnn.swin_utils.build import make_swin
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights, vgg16, VGG16_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def save_image(image, path, ground_truth=[], prediction=[], scores=[], threshold=0.5, orig_size=None):
    fig, ax = plt.subplots(1)
    ax.axis('off')
    ax.imshow(image)

    for i, box in enumerate(prediction):
        score = scores[i].item()
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    for box in ground_truth:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)

    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)

    if orig_size is not None:
        img = cv2.imread(path)
        img = cv2.resize(img, (orig_size[1], orig_size[0]))
        cv2.imwrite(path, img)


def gethyperparameters(config):
    name = config['inf_name']
    parts = name.split('_')
    config['model']['model_type'] = parts[0]
    if config['model']['model_type'] == "swin":
        config['model']['backbone_name'] = "_".join(parts[1:6]).split('-')[1]
        config['model']['fpn'] = parts[6].split('-')[1] == "True"
        config['model']['lora'] = parts[7].split('-')[1] == "True"
    else:
        config['model']['lora'] = parts[1].split('-')[1] == "True"
    return config


def build_model(config):
    if config['model']['model_type'] == "swin":
        model = make_swin(config)
    elif config['model']['model_type'] == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(num_classes=config['model']['nclasses'],
                                        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                                        trainable_backbone_layers=5,
                                        max_size=config['size'], min_size=config['size'])
    elif config['model']['model_type'] == "custom":
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )
        model = FasterRCNN(
            backbone=backbone,
            num_classes=config['model']['nclasses'],
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['model_type']}. Choose 'swin', 'fasterrcnn' or 'custom'.")
    return model


def build_transforms(transform_cfg):
    transforms = []
    for t in transform_cfg:
        t = dict(t)
        name = t.pop('name')
        cls = getattr(A, name, None)
        if cls is None and name == 'ToTensorV2':
            cls = ToTensorV2
        if cls:
            transforms.append(cls(**t))
    return transforms


def make_transforms(config):
    test_cfg = config.get('test_transforms', {})
    return A.Compose(
        build_transforms(test_cfg),
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )


def convert_to_8bit(image_16bit):
    min_val = image_16bit.min()
    max_val = image_16bit.max()
    if max_val > min_val:
        return ((image_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return np.zeros_like(image_16bit, dtype=np.uint8)
