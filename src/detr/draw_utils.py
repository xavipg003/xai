import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def save_image_with_boxes(image, pred_boxes, ground_truth, output_path):
    if image.dim() == 4:
        image = image.squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)

    image = image * std + mean

    image= image.cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    alto_img, ancho_img, _ = image.shape
    fig, ax = plt.subplots(1)
    ax.axis('off')
    
    ax.imshow(image)
    for caja in pred_boxes:
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = caja
        
        x_pixel = xmin_norm * ancho_img
        y_pixel = ymin_norm * alto_img
        w_pixel = (xmax_norm - xmin_norm) * ancho_img
        h_pixel = (ymax_norm - ymin_norm) * alto_img
        
        rect = patches.Rectangle(
            (x_pixel, y_pixel), w_pixel, h_pixel,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    for caja in ground_truth:
        x_norm, y_norm, w_norm, h_norm = caja
        
        x_pixel = x_norm * ancho_img
        y_pixel = y_norm * alto_img
        w_pixel = w_norm * ancho_img
        h_pixel = h_norm * alto_img
        
        rect = patches.Rectangle(
            (x_pixel-w_pixel/2, y_pixel-h_pixel/2), w_pixel, h_pixel,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)

    legend_handles = [
        patches.Patch(edgecolor='green', facecolor='none', linewidth=2, label='Ground Truth'),
        patches.Patch(edgecolor='red',   facecolor='none', linewidth=2, label='Prediction'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              framealpha=0.7, fontsize=8, handlelength=1.5)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    
    plt.close(fig)
    print(f"Saved in: {output_path}")
