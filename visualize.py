import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from pathlib import Path
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import _utils as det_utils

from src.dataset.dataset import MedicalDataset


def visualize_all():
    device = torch.device('cpu')
    # .parent берет ту папку, в которой прямо сейчас лежит скрипт
    project_root = Path(__file__).resolve().parent

    base_dir = project_root / "data" / "raw" / "images" / "train"
    labels_dir = project_root / "data" / "raw" / "labels" / "train"
    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    image, target = dataset[0]  # Берем снимок №1
    image_uint8 = (image * 255).to(torch.uint8)
    num_classes = 3

    models_config = {
        "SSD": ("ssd_weights.pth",
                lambda: torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")),
        "Faster R-CNN": ("faster_rcnn_weights.pth",
                         lambda: torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")),
        "RetinaNet": ("retinanet_weights.pth",
                      lambda: torchvision.models.detection.retinanet_resnet50_fpn(weights_backbone="DEFAULT",
                                                                                  num_classes=num_classes)),
        "FCOS": ("fcos_weights.pth", lambda: torchvision.models.detection.fcos_resnet50_fpn(weights_backbone="DEFAULT",
                                                                                            num_classes=num_classes))
    }

    # Создаем красивую сетку 2x2 для отчета
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (model_name, (file_name, init_func)) in enumerate(models_config.items()):
        weights_path = project_root / file_name
        ax = axes[idx]
        ax.axis('off')

        if not weights_path.exists():
            ax.set_title(f"{model_name}\n(Веса не найдены)", fontsize=16)
            continue

        model = init_func()

        if model_name == "SSD":
            in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        elif model_name == "Faster R-CNN":
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        with torch.no_grad():
            prediction = model([image])

        pred_boxes = prediction[0]['boxes']
        pred_scores = prediction[0]['scores']
        keep = pred_scores > 0.3  # Оставляем только уверенные предсказания
        pred_boxes = pred_boxes[keep]

        result_image = draw_bounding_boxes(image_uint8, boxes=pred_boxes, colors="red", width=3)
        ax.imshow(result_image.permute(1, 2, 0).numpy())
        ax.set_title(f"{model_name}", fontsize=18, fontweight='bold')

    plt.tight_layout()
    save_path = project_root / "results" / "plots" / "all_models_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Отличная работа! Сравнительный коллаж сохранен в: {save_path}")


if __name__ == '__main__':
    visualize_all()