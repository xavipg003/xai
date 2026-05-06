import os
import numpy as np
import torch
from matplotlib import pyplot as plt


def compute_metrics(heatmap, model, image_tensor, gt_boxes, device, output_dir, idx, method_name,
                    n_deletion_steps=20):
    """
    Computes three XAI evaluation metrics and returns them as a dict.

    - Pointing Game : fraction of GT boxes whose interior contains the saliency argmax.
    - IoU           : overlap between the top-5% saliency mask and the GT box mask.
    - Deletion AUC  : area under the curve obtained by progressively zeroing the most
                      salient pixels and recording the model's max detection score.
                      Lower AUC → the explanation captured truly important pixels.
    """
    H, W = heatmap.shape
    pos_heatmap = np.maximum(heatmap, 0)

    # Pointing Game
    if pos_heatmap.max() > 0:
        max_y, max_x = np.unravel_index(pos_heatmap.argmax(), pos_heatmap.shape)
    else:
        max_y, max_x = 0, 0

    hits = 0
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box
        if xmin <= max_x <= xmax and ymin <= max_y <= ymax:
            hits += 1
    pointing_score = hits / len(gt_boxes) if len(gt_boxes) > 0 else 0.0

    # IoU (top-5% positive saliency vs. GT mask)
    gt_mask = np.zeros((H, W), dtype=bool)
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        gt_mask[ymin:ymax, xmin:xmax] = True

    if pos_heatmap.max() > 0:
        threshold = np.percentile(pos_heatmap[pos_heatmap > 0], 95)
        sal_mask = pos_heatmap >= threshold
    else:
        sal_mask = np.zeros((H, W), dtype=bool)

    intersection = (sal_mask & gt_mask).sum()
    union = (sal_mask | gt_mask).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    # Deletion Curve
    flat_importance = pos_heatmap.flatten()
    sorted_indices = np.argsort(flat_importance)[::-1]
    n_pixels = len(flat_importance)

    percentages = np.linspace(0, 1, n_deletion_steps + 1)
    deletion_scores = []

    with torch.no_grad():
        for pct in percentages:
            n_delete = int(pct * n_pixels)
            img_mod = image_tensor.clone().to(device)
            if n_delete > 0:
                indices = sorted_indices[:n_delete]
                rows, cols = indices // W, indices % W
                img_mod[:, rows, cols] = 0.0
            outputs = model([img_mod])
            score = outputs[0]['scores'].max().item() if len(outputs[0]['scores']) > 0 else 0.0
            deletion_scores.append(score)

    deletion_auc = float(np.trapz(deletion_scores, percentages))

    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(percentages * 100, deletion_scores, marker='o', markersize=3)
    ax.set_xlabel("% pixels deleted (most salient first)")
    ax.set_ylabel("Max detection score")
    ax.set_title(f"Deletion Curve [{method_name}]  —  AUC={deletion_auc:.3f}")
    ax.set_ylim(0, 1)
    plt.savefig(os.path.join(img_out_dir, f"{idx}_{method_name}_deletion_curve.png"), bbox_inches='tight')
    plt.close(fig)

    return {'pointing_game': pointing_score, 'iou': iou, 'deletion_auc': deletion_auc}
