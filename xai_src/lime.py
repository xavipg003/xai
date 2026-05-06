import os
import numpy as np
import torch
from src.faster_rcnn.pt_lightning.utils import save_image


def run_lime(Lmodel, inputs, output_dir, idx, device, threshold=0.5):
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

    segments = explanation.segments
    heatmap = np.zeros(segments.shape, dtype=float)
    for seg_id, weight in explanation.local_exp.get(explanation.top_labels[0], []):
        heatmap[segments == seg_id] = weight
    return heatmap
