import random
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from src.faster_rcnn.pt_lightning.utils import build_model, gethyperparameters, save_image
from src.faster_rcnn.pt_lightning.classes import MyDataModule, CustomModel


def inference(config):
    config = gethyperparameters(config)
    output_dir = config['paths']['output_images']
    model_path = config['paths']['model_path']
    inf_name = config['inf_name']

    datamodule = MyDataModule(config)
    datamodule.setup()

    model = build_model(config)
    Lmodel = CustomModel(config, model)
    state_dict = torch.load(f'{model_path}/{inf_name}')['state_dict']
    Lmodel.load_state_dict(state_dict)
    Lmodel.eval()

    test_dataset = datamodule.test_dataset
    idx = random.randint(0, len(test_dataset) - 1)
    image, target, orig_size = test_dataset[idx]

    with torch.no_grad():
        prediction = Lmodel.model([image])

    save_image(
        image.permute(1, 2, 0).numpy(),
        f"{output_dir}/output.png",
        ground_truth=target['boxes'].numpy(),
        prediction=prediction[0]['boxes'].numpy(),
        scores=prediction[0]['scores'].numpy(),
        threshold=0.75,
        orig_size=orig_size
    )
