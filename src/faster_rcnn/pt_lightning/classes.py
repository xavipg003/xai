import torch
import lightning as L
from torch.utils.data import Dataset

import os
from PIL import Image, ImageFile

import json
import numpy as np

from pathlib import Path
from src.faster_rcnn.pt_lightning.utils import make_transforms, convert_to_8bit


class CustomModel(L.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class AlbumentationsImageDataset(Dataset):
    def __init__(self, config, img_dir, file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.file = file
        self.config = config

        with open(self.file, 'r') as f:
            self.data = json.load(f)

        self.img_names = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        data = self.data
        img_name = self.img_names[idx]

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = np.array(Image.open(img_path))
        except:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = np.array(Image.open(img_path))

        image = convert_to_8bit(image)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        orig_size = image.shape[:2]

        img_id = [img['id'] for img in data['images'] if img['file_name'] == img_name][0]
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        boxes = []
        label_list = []

        for ann in annotations:
            x_min, y_min, w, h = ann['bbox']
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            label_list.append(ann['category_id'] + 1)

        if len(annotations) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            labels = label_list

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4), dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return image, target, orig_size


class MyDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.test_image_dir = Path(config['paths']['test'])
        self.label_file = Path(config['paths']['labels'])
        self.transform_test = make_transforms(config)
        self.config = config
        self.nworkers = int(os.getenv("OMP_NUM_THREADS", 31))

    def setup(self, stage=None):
        self.test_dataset = AlbumentationsImageDataset(
            self.config,
            img_dir=self.test_image_dir,
            file=self.label_file,
            transform=self.transform_test
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)), num_workers=self.nworkers
        )
