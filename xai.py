import argparse
import os
import random
import warnings
import numpy as np
import torch
from omegaconf import OmegaConf
from src.faster_rcnn.pt_lightning.utils import build_model, gethyperparameters, save_image
from src.faster_rcnn.pt_lightning.classes import MyDataModule, CustomModel

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("CUDA not available, falling back to CPU.")
    return torch.device("cpu")


def load_model_and_data(config, device):
    config = gethyperparameters(config)
    model_path = config['paths']['model_path']
    inf_name = config['inf_name']

    datamodule = MyDataModule(config)
    datamodule.setup()

    model = build_model(config)
    Lmodel = CustomModel(config, model)
    state_dict = torch.load(f'{model_path}/{inf_name}', map_location=device)['state_dict']
    Lmodel.load_state_dict(state_dict)
    Lmodel.eval()
    Lmodel.model.to(device)

    return Lmodel, datamodule.test_dataset, config


def run_gradcam(Lmodel, inputs, output_dir, idx, device):
    from pytorch_grad_cam import HiResCAM
    from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = Lmodel.model
    orig_size = inputs[2]

    with torch.no_grad():
        preds = model([inputs[0].to(device)])

    boxes = preds[0]["boxes"]
    labels = preds[0]["labels"]
    scores = preds[0]["scores"]

    mask = scores > 0.75
    boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

    if len(boxes) == 0:
        print(f"  No detections above threshold for image {idx}, skipping GradCAM.")
        return

    layers_list = [
        [model.backbone.body.layer1[-1]],
        [model.backbone.body.layer2[-1]],
        [model.backbone.body.layer3[-1]],
        [model.backbone.body.layer4[-1]],
        [model.backbone.fpn.inner_blocks[0]],
        [model.backbone.fpn.inner_blocks[1]],
        [model.backbone.fpn.inner_blocks[2]],
        [model.backbone.fpn.inner_blocks[-1]],
        [model.backbone.fpn.layer_blocks[0]],
        [model.backbone.fpn.layer_blocks[1]],
        [model.backbone.fpn.layer_blocks[2]],
        [model.backbone.fpn.layer_blocks[3]],
        [model.backbone.body.layer3[-1], model.backbone.body.layer4[-1]],
        [model.backbone.fpn.layer_blocks[0], model.backbone.fpn.layer_blocks[2]],
    ]

    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)

    for i, target_layers in enumerate(layers_list):
        with HiResCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=inputs[0].unsqueeze(0).to(device), targets=targets)

        cam_image = show_cam_on_image(inputs[0].permute(1, 2, 0).numpy(), grayscale_cam[0], use_rgb=True)
        save_image(cam_image, os.path.join(img_out_dir, f"{idx}_gradcam_{i}.png"),
                   ground_truth=inputs[1]['boxes'].cpu().numpy(),
                   prediction=boxes.cpu(),
                   scores=scores.cpu(),
                   threshold=0.75, orig_size=orig_size)


def run_lime(Lmodel, inputs, output_dir, idx, device):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    orig_size = inputs[2]
    image = inputs[0].permute(1, 2, 0).numpy()

    def predict(images):
        scores = []
        with torch.no_grad():
            for img in images:
                img = torch.tensor(img).permute(2, 0, 1).to(device)
                outputs = Lmodel.model([img])
                scores.append([outputs[0]['scores'].max().item() if len(outputs[0]['scores']) > 0 else 0.0])
        return np.array(scores)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, predict, top_labels=1, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False
    )
    result = mark_boundaries(temp, mask)

    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)
    save_image(result, os.path.join(img_out_dir, f"{idx}_lime.png"),
               ground_truth=inputs[1]['boxes'].cpu().numpy(), orig_size=orig_size)


def run_shap(Lmodel, inputs, output_dir, idx, device):
    import shap
    import cv2
    from matplotlib import pyplot as plt

    orig_size = inputs[2]
    image = inputs[0].permute(1, 2, 0).numpy()

    def predict(images):
        scores = []
        with torch.no_grad():
            for img in images:
                img = torch.tensor(img).permute(2, 0, 1).to(device)
                outputs = Lmodel.model([img])
                scores.append([outputs[0]['scores'].max().item() if len(outputs[0]['scores']) > 0 else 0.0])
        return np.array(scores)

    masker = shap.maskers.Image("blur(128,128)", image.shape)
    explainer = shap.Explainer(predict, masker)
    shap_values = explainer(image[np.newaxis, ...], max_evals=1000)

    shap_vals = shap_values.values.squeeze()
    heatmap = shap_vals.sum(axis=-1)

    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)
    out_path = os.path.join(img_out_dir, f"{idx}_shap_heatmap.png")

    fig, ax = plt.subplots()
    ax.imshow(image)
    abs_max = np.abs(heatmap).max()
    ax.imshow(heatmap, cmap='RdBu', alpha=0.6, vmin=-abs_max, vmax=abs_max)
    ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    if orig_size is not None:
        img = cv2.imread(out_path)
        img = cv2.resize(img, (orig_size[1], orig_size[0]))
        cv2.imwrite(out_path, img)

    save_image(image, os.path.join(img_out_dir, f"{idx}_shap_gt.png"),
               ground_truth=inputs[1]['boxes'].cpu().numpy(), orig_size=orig_size)


METHODS = {
    'gradcam': run_gradcam,
    'lime': run_lime,
    'shap': run_shap,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI explainability pipeline for object detector")
    parser.add_argument('--method', choices=['gradcam', 'lime', 'shap', 'all'], default='gradcam',
                        help="Explainability method (default: gradcam)")
    parser.add_argument('--mode', choices=['single', 'dataset'], default='single',
                        help="Run on a single image or the full test dataset (default: single)")
    parser.add_argument('--index', type=int, default=None,
                        help="Image index for single mode (random if not specified)")
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                        help="Device to use: 'cuda' or 'cpu'. Auto-detects if not specified.")
    args = parser.parse_args()

    device = get_device(force_cpu=(args.device == 'cpu'))
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load("config/config_faster.yaml")
    Lmodel, test_dataset, config = load_model_and_data(config, device)
    output_dir = config['paths']['output_images']

    methods = list(METHODS.keys()) if args.method == 'all' else [args.method]

    if args.mode == 'single':
        idx = args.index if args.index is not None else random.randint(0, len(test_dataset) - 1)
        inputs = test_dataset[idx]
        print(f"Running {methods} on image index {idx}")
        for method in methods:
            print(f"  [{method}]")
            METHODS[method](Lmodel, inputs, output_dir, idx, device)
    else:
        for idx in range(len(test_dataset)):
            inputs = test_dataset[idx]
            print(f"Image {idx + 1}/{len(test_dataset)}")
            for method in methods:
                print(f"  [{method}]")
                METHODS[method](Lmodel, inputs, output_dir, idx, device)
