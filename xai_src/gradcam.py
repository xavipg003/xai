import os
import torch
from src.faster_rcnn.pt_lightning.utils import save_image

def swin_reshape_transform(tensor):
    # (B, H, W, C) -> (B, C, H, W)
    return tensor.permute(0, 3, 1, 2)


def run_gradcam(Lmodel, inputs, output_dir, idx, device, threshold=0.5):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = Lmodel.model
    orig_size = inputs[2]

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

    # layers_list = [
    #     # Stages del Swin (necesitan reshape_transform)
    #     [model.backbone.swin.layers[0].blocks[-1].norm2],
    #     [model.backbone.swin.layers[1].blocks[-1].norm2],
    #     [model.backbone.swin.layers[2].blocks[-1].norm2],
    #     [model.backbone.swin.layers[3].blocks[-1].norm2],

    #     # FPN inner_blocks (NO necesitan reshape_transform)
    #     [model.backbone.fpn.inner_blocks[0]],
    #     [model.backbone.fpn.inner_blocks[1]],
    #     [model.backbone.fpn.inner_blocks[2]],
    #     [model.backbone.fpn.inner_blocks[-1]],

    #     # FPN layer_blocks (NO necesitan reshape_transform)
    #     [model.backbone.fpn.layer_blocks[0]],
    #     [model.backbone.fpn.layer_blocks[1]],
    #     [model.backbone.fpn.layer_blocks[2]],
    #     [model.backbone.fpn.layer_blocks[3]],

    #     # Combinaciones
    #     [model.backbone.swin.layers[2].blocks[-1].norm2,
    #     model.backbone.swin.layers[3].blocks[-1].norm2],   # con reshape_transform
    #     [model.backbone.fpn.layer_blocks[0],
    #     model.backbone.fpn.layer_blocks[2]],               # sin reshape_transform
    # ]

    # Each entry: (target_layers, needs_reshape)
    # Swin blocks output (B, H, W, C) → need reshape_transform
    # FPN blocks output (B, C, H, W)  → no reshape needed
    layers_list = [
        ([model.backbone.swin.layers[0].blocks[-1]], True),
        ([model.backbone.swin.layers[1].blocks[-1]], True),
        ([model.backbone.swin.layers[2].blocks[-1]], True),
        ([model.backbone.swin.layers[3].blocks[-1]], True),
        ([model.backbone.fpn.inner_blocks[0]], False),
        ([model.backbone.fpn.inner_blocks[1]], False),
        ([model.backbone.fpn.inner_blocks[2]], False),
        ([model.backbone.fpn.inner_blocks[-1]], False),

        ([model.backbone.fpn.layer_blocks[0]], False),
        ([model.backbone.fpn.layer_blocks[1]], False),
        ([model.backbone.fpn.layer_blocks[2]], False),
        ([model.backbone.fpn.layer_blocks[3]], False),

        ([model.backbone.swin.layers[2].blocks[-1],
          model.backbone.swin.layers[3].blocks[-1]], True),

        ([model.backbone.fpn.layer_blocks[0],
          model.backbone.fpn.layer_blocks[2]], False),
    ]

    # Index of fpn.inner_blocks[-1] — used as the canonical heatmap for metrics
    METRICS_LAYER_IDX = 7

    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    img_out_dir = os.path.join(output_dir, str(idx))
    os.makedirs(img_out_dir, exist_ok=True)

    metrics_heatmap = None
    for i, (target_layers, needs_reshape) in enumerate(layers_list):
        reshape_fn = swin_reshape_transform if needs_reshape else None
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
