import torch
from torchmetrics.detection import MeanAveragePrecision
from datasets import load_from_disk
from src.detr.utils import transform, make_transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = load_from_disk(config['paths']['data'])

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",do_resize=True,do_pad=True)

    model_path = config['paths']['model_path']

    transform_train, transform_test = make_transforms(config)
    test_dataset = dataset["test"].with_transform(lambda x: transform(x, processor, aug_transform=transform_test))

    model = DetrForObjectDetection.from_pretrained(model_path,ignore_mismatched_sizes=True,
            num_labels=1,)

    model.to(device)

    metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=None,  
            class_metrics=False,    
            average='macro'
        )

    model.eval()

    with torch.no_grad():
        for i in range(len(test_dataset)):
            inputs=test_dataset[i]
            image=inputs['pixel_values']
            
            pixel_values_gpu = inputs['pixel_values'].unsqueeze(0).to(device)
            pixel_mask_gpu = inputs['pixel_mask'].unsqueeze(0).to(device)
            
            outputs = model(pixel_values=pixel_values_gpu, pixel_mask=pixel_mask_gpu)
            
            outputs = processor.post_process_object_detection(outputs, threshold=0.8)

            if image.dim() == 4:
                image = image.squeeze(0)

            image= image.cpu().numpy() 
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
            alto_img, ancho_img, _ = image.shape
            image=inputs['pixel_values'] 
            
            scores = outputs[0]["scores"].detach().cpu().numpy()
            outputs = outputs[0]["boxes"].detach().cpu().numpy() 
            
            final_preds=[]
            for pred in outputs:
                xmin_norm, ymin_norm, xmax_norm, ymax_norm = pred
            
                xmin_pixel = xmin_norm * ancho_img
                ymin_pixel = ymin_norm * alto_img
                xmax_pixel = xmax_norm * ancho_img
                ymax_pixel = ymax_norm* alto_img
                final_preds.append([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel])

            inputs=inputs['labels']['boxes'].numpy()
            final_inputs=[]
            for label in inputs:
                x_norm, y_norm, w_norm, h_norm = label
            
                x_pixel = x_norm * ancho_img
                y_pixel = y_norm * alto_img
                w_pixel = w_norm * ancho_img
                h_pixel = h_norm * alto_img
                final_inputs.append([x_pixel-w_pixel/2, y_pixel-h_pixel/2, x_pixel + w_pixel/2, y_pixel + h_pixel/2])
            
            final_inputs=np.array(final_inputs).tolist()
            final_preds=np.array(final_preds).tolist()

            final_preds=[{'boxes': torch.tensor(final_preds), 
                        'scores': torch.tensor(scores), 
                        'labels': torch.ones(len(final_preds), dtype=torch.int64)}]
            final_inputs=[{'boxes': torch.tensor(final_inputs), 
                        'labels': torch.ones(len(final_inputs), dtype=torch.int64)}]
            
            metric.update(final_preds, final_inputs)

            print(f"Processed image {i+1}/{len(test_dataset)}")

    metrics_dict = metric.compute()

    map= metrics_dict["map"]
    map50 = metrics_dict["map_50"]
    map75 = metrics_dict["map_75"]
    mar = metrics_dict["mar_100"]
    print(f"mAP: {map}")
    print(f"mAP50: {map50}")
    print(f"mAR: {mar}")
    print(f"mAP75: {map75}")

