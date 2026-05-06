import argparse
import csv
import os
import random
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf

from src.faster_rcnn.pt_lightning.utils import build_model, gethyperparameters
from src.faster_rcnn.pt_lightning.classes import MyDataModule, CustomModel
from xai_src.gradcam import run_gradcam
from xai_src.lime import run_lime
from xai_src.shap_xai import run_shap
from xai_src.ig import run_integrated_gradients
from xai_src.metrics import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)

METHODS = {
    'gradcam': run_gradcam,
    'lime': run_lime,
    'shap': run_shap,
    'ig': run_integrated_gradients,
}


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


def save_predictions_txt(output_dir, idx, image_name, orig_size, gt_boxes, gt_labels,
                         pred_boxes, pred_labels, pred_scores, threshold, model_name=""):
    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(img_out_dir, f"{idx}_predictions_{i}.txt")):
        i += 1
    txt_path = os.path.join(img_out_dir, f"{idx}_predictions_{i}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Model         : {model_name}\n")
        f.write(f"Image index   : {idx}\n")
        f.write(f"Image file    : {image_name}\n")
        f.write(f"Original size : H={orig_size[0]}  W={orig_size[1]}\n")
        f.write(f"Threshold     : {threshold}\n")

        f.write("\n--- Ground Truth ---\n")
        if len(gt_boxes) == 0:
            f.write("  (none)\n")
        else:
            for i, (box, lbl) in enumerate(zip(gt_boxes, gt_labels)):
                f.write(f"  [{i}] label={int(lbl)}  "
                        f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")

        f.write(f"\n--- Predictions (all, sorted by score) ---\n")
        if len(pred_boxes) == 0:
            f.write("  (none)\n")
        else:
            order = pred_scores.argsort()[::-1]
            for i, k in enumerate(order):
                marker = "✓" if pred_scores[k] >= threshold else "✗"
                f.write(f"  [{i}] {marker}  label={int(pred_labels[k])}  "
                        f"score={pred_scores[k]:.4f}  "
                        f"box=[{pred_boxes[k][0]:.1f}, {pred_boxes[k][1]:.1f}, "
                        f"{pred_boxes[k][2]:.1f}, {pred_boxes[k][3]:.1f}]\n")

        n_above = int((pred_scores >= threshold).sum())
        f.write(f"\nDetections above threshold: {n_above}/{len(pred_boxes)}\n")
    print(f"  Predictions saved → {txt_path}")


def process_image(Lmodel, inputs, idx, methods, output_dir, device, threshold, metrics_log, compute,
                  image_name="", model_name=""):
    gt_boxes = inputs[1]['boxes'].cpu().numpy()
    gt_labels = inputs[1]['labels'].cpu().numpy()
    orig_size = inputs[2]

    with torch.no_grad():
        preds = Lmodel.model([inputs[0].to(device)])
    pred_boxes  = preds[0]['boxes'].cpu().numpy()
    pred_labels = preds[0]['labels'].cpu().numpy()
    pred_scores = preds[0]['scores'].cpu().numpy()

    save_predictions_txt(output_dir, idx, image_name, orig_size,
                         gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, threshold,
                         model_name=model_name)

    for method in methods:
        print(f"  [{method}]")
        heatmap = METHODS[method](Lmodel, inputs, output_dir, idx, device, threshold)
        if compute:
            if heatmap is None:
                print(f"    Metrics skipped (no heatmap).")
                continue
            m = compute_metrics(heatmap, Lmodel.model, inputs[0], gt_boxes,
                                device, output_dir, idx, method)
            metrics_log[method].append({'idx': idx, **m})
            print(f"    pointing_game={m['pointing_game']:.3f}  "
                  f"iou={m['iou']:.3f}  "
                  f"deletion_auc={m['deletion_auc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI explainability pipeline for object detector")
    parser.add_argument('--method', choices=['gradcam', 'lime', 'shap', 'ig', 'all'], default='gradcam')
    parser.add_argument('--mode', choices=['single', 'dataset'], default='single')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--metrics', action='store_true',
                        help="Compute XAI metrics: Pointing Game, IoU, Deletion Curve AUC")
    args = parser.parse_args()

    device = get_device(force_cpu=(args.device == 'cpu'))
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load("config/config_faster.yaml")
    Lmodel, test_dataset, config = load_model_and_data(config, device)
    output_dir = config['paths']['output_images']
    model_name = config['inf_name']

    methods = list(METHODS.keys()) if args.method == 'all' else [args.method]
    metrics_log = {m: [] for m in methods}

    if args.mode == 'single':
        idx = args.index if args.index is not None else random.randint(0, len(test_dataset) - 1)
        print(f"Running {methods} on image index {idx}")
        process_image(Lmodel, test_dataset[idx], idx, methods, output_dir, device, args.threshold,
                      metrics_log, args.metrics, image_name=test_dataset.img_names[idx],
                      model_name=model_name)
    else:
        for idx in range(len(test_dataset)):
            print(f"Image {idx + 1}/{len(test_dataset)}")
            process_image(Lmodel, test_dataset[idx], idx, methods, output_dir, device, args.threshold,
                          metrics_log, args.metrics, image_name=test_dataset.img_names[idx],
                          model_name=model_name)

    if args.metrics and args.mode == 'dataset':
        for method, rows in metrics_log.items():
            if not rows:
                continue
            csv_path = os.path.join(output_dir, f"metrics_{method}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['idx', 'pointing_game', 'iou', 'deletion_auc'])
                writer.writeheader()
                writer.writerows(rows)

            pg_mean  = np.mean([r['pointing_game'] for r in rows])
            iou_mean = np.mean([r['iou'] for r in rows])
            del_mean = np.mean([r['deletion_auc'] for r in rows])
            print(f"\n[{method}] Dataset averages — "
                  f"pointing_game={pg_mean:.3f}  iou={iou_mean:.3f}  deletion_auc={del_mean:.3f}")
            print(f"  Saved per-image metrics to {csv_path}")
