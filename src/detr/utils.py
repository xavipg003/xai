import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

def make_transforms(config):
    train_cfg = config.get('train_transforms', {})
    test_cfg = config.get('test_transforms', {})

    # Construir transforms
    transform_train = A.Compose(
        build_transforms(train_cfg),
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
    )
    transform_test = A.Compose(
        build_transforms(test_cfg),
        bbox_params=A.BboxParams(format='coco', label_fields=['labels'])
    )
    return transform_train, transform_test

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

def transform(example_batch, processor, aug_transform=None):
    transformed_batch={'images':[], 'objects':[]}
    if aug_transform:
        for img, objects in zip(example_batch["image"], example_batch["objects"]):
            valid=False
            while not valid:
                transformed = aug_transform(
                    image=np.array(img),
                    bboxes=objects['bbox'],
                    labels=objects['category']
                )
                if len(transformed['bboxes'])>0:
                    valid=True
            
            transformed_batch['images'].append(transformed['image'])
            transformed_batch['objects'].append({
                'bbox': [transformed['bboxes'][0]],
                'category': [int(label) for label in transformed['labels']],
                'image_id': objects['image_id']
            })
        batch=transformed_batch
    
    images = batch["images"]

    annotations = [{
        "image_id": ex["image_id"],
        "annotations": [
            {
                "image_id": ex["image_id"],
                "bbox": box,
                "category_id": cat,
                "area": box[2] * box[3],  # width * height
                "iscrowd": 0
            }
            for box, cat in zip(ex["bbox"], ex["category"])
        ],
    } for ex in batch["objects"]]

    inputs = processor(images=images, annotations=annotations, return_tensors="pt")

    return inputs

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def get_name(config):
    name = (
        f"detr_bs-{config['training']['batch_size']}"
        f"_lr-{config['training']['learning_rate']}"
    )
    return name

