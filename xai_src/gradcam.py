import os
import torch
from src.faster_rcnn.pt_lightning.utils import save_image


def swin_reshape_transform(tensor):
    # (B, H, W, C) -> (B, C, H, W)
    return tensor.permute(0, 3, 1, 2)



def list_gradcam_layers(model):
    print("\nAvailable layers for GradCAM:")
    for name, module in model.named_modules():
        print(f"  {name:80s}  [{module.__class__.__name__}]")


def run_gradcam(Lmodel, inputs, output_dir, idx, device, threshold=0.5, dry_run=False):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = Lmodel.model
    orig_size = inputs[2]

    if dry_run:
        list_gradcam_layers(model)
        return None

    with torch.no_grad():
        preds = model([inputs[0].to(device)])

    boxes = preds[0]["boxes"]
    labels = preds[0]["labels"]
    scores = preds[0]["scores"]

    mask = scores > threshold
    boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

    if len(boxes) == 0:
        print(f"  No detections above threshold for image {idx}, skipping GradCAM.")
        return None

    # Each entry: (target_layers, reshape_fn or None)
    # Swin blocks output (B, H, W, C) → swin_reshape_transform
    # FPN blocks output (B, C, H, W)  → None
    layers_list = [
        ([model.backbone.swin.layers[0].blocks[-1]], swin_reshape_transform),
        ([model.backbone.swin.layers[1].blocks[-1]], swin_reshape_transform),
        ([model.backbone.swin.layers[2].blocks[-1]], swin_reshape_transform),
        ([model.backbone.swin.layers[3].blocks[-1]], swin_reshape_transform),
        ([model.backbone.fpn.inner_blocks[0]], None),
        ([model.backbone.fpn.inner_blocks[1]], None),
        ([model.backbone.fpn.inner_blocks[2]], None),
        ([model.backbone.fpn.inner_blocks[-1]], None),

        ([model.backbone.fpn.layer_blocks[0]], None),
        ([model.backbone.fpn.layer_blocks[1]], None),
        ([model.backbone.fpn.layer_blocks[2]], None),
        ([model.backbone.fpn.layer_blocks[3]], None),

        ([model.backbone.swin.layers[2].blocks[-1],
          model.backbone.swin.layers[3].blocks[-1]], swin_reshape_transform),

        ([model.backbone.fpn.layer_blocks[0],
          model.backbone.fpn.layer_blocks[2]], None),
    ]

    # Index of fpn.inner_blocks[2] — used as the canonical heatmap for metrics
    METRICS_LAYER_IDX = 6

    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)

    metrics_heatmap = None
    for i, (target_layers, reshape_fn) in enumerate(layers_list):
        with GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_fn) as cam:
            grayscale_cam = cam(input_tensor=inputs[0].unsqueeze(0).to(device), targets=targets)

        if i == METRICS_LAYER_IDX:
            metrics_heatmap = grayscale_cam[0]

        cam_image = show_cam_on_image(inputs[0].permute(1, 2, 0).numpy(), grayscale_cam[0], use_rgb=True)
        save_image(cam_image, os.path.join(img_out_dir, f"{idx}_gradcam_{i}.png"),
                   ground_truth=inputs[1]['boxes'].cpu().numpy(),
                   prediction=boxes.cpu(),
                   scores=scores.cpu(),
                   threshold=threshold, orig_size=orig_size)

    return metrics_heatmap
