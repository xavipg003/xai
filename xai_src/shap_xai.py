import os
import numpy as np
import torch
from src.faster_rcnn.pt_lightning.utils import save_image


def run_shap(Lmodel, inputs, output_dir, idx, device, threshold=0.5):
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

    return heatmap
