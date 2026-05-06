import os
import numpy as np
import torch
from src.faster_rcnn.pt_lightning.utils import save_image


def run_integrated_gradients(Lmodel, inputs, output_dir, idx, device, threshold=0.5):
    import cv2
    from matplotlib import pyplot as plt

    n_steps = 50
    model = Lmodel.model
    orig_size = inputs[2]
    image_tensor = inputs[0].to(device)
    image_np = inputs[0].permute(1, 2, 0).numpy()

    with torch.no_grad():
        preds = model([image_tensor])

    det_scores = preds[0]["scores"]
    boxes = preds[0]["boxes"]
    labels = preds[0]["labels"]
    mask = det_scores > threshold
    boxes, labels, det_scores = boxes[mask], labels[mask], det_scores[mask]

    if len(boxes) == 0:
        print(f"  No detections above threshold for image {idx}, skipping Integrated Gradients.")
        return

    baseline = torch.zeros_like(image_tensor)
    grads_sum = torch.zeros_like(image_tensor)
    valid_steps = 0

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp_input = (baseline + alpha * (image_tensor - baseline)).detach().requires_grad_(True)
        outputs = model([interp_input])

        if len(outputs[0]['scores']) == 0:
            continue

        target = outputs[0]['scores'].sum()
        model.zero_grad()
        target.backward()

        if interp_input.grad is not None:
            grads_sum += interp_input.grad.detach()
            valid_steps += 1

    if valid_steps == 0:
        print(f"  No gradients computed for image {idx}, skipping Integrated Gradients.")
        return

    integrated_grads = (grads_sum / valid_steps) * (image_tensor - baseline)
    attr = integrated_grads.cpu().numpy()
    heatmap = attr.sum(axis=0)
    heatmap_vis = heatmap.copy()

    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)
    out_path = os.path.join(img_out_dir, f"{idx}_ig_heatmap.png")

    fig, ax = plt.subplots()
    ax.imshow(image_np)
    abs_max = np.percentile(np.abs(heatmap_vis), 99)
    if abs_max > 0:
        from scipy.ndimage import gaussian_filter
        heatmap_vis = gaussian_filter(heatmap_vis, sigma=8)
        ax.imshow(heatmap_vis, cmap='RdBu', alpha=0.6, vmin=-abs_max, vmax=abs_max)
    ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    if orig_size is not None:
        img = cv2.imread(out_path)
        img = cv2.resize(img, (orig_size[1], orig_size[0]))
        cv2.imwrite(out_path, img)

    save_image(image_np, os.path.join(img_out_dir, f"{idx}_ig_gt.png"),
               ground_truth=inputs[1]['boxes'].cpu().numpy(),
               prediction=boxes.cpu(),
               scores=det_scores.cpu(),
               threshold=threshold, orig_size=orig_size)

    return heatmap
