import os
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

from src.dataset.dataset import MedicalDataset


def evaluate_all_comprehensive():
    device = torch.device('cpu')
    project_root = Path(__file__).resolve().parents[2]

    # 1. Загрузка тестовых данных
    base_dir = project_root / "data" / "raw" / "images" / "train"
    labels_dir = project_root / "data" / "raw" / "labels" / "train"
    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    num_classes = 3

    # Таблица для хранения всех результатов
    results_df = pd.DataFrame(columns=[
        'Model', 'Precision', 'Recall', 'F1-score', 'mAP', 'mAP@50', 'mAP@50:95'
    ])

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

    print("=== СТАРТ КОМПЛЕКСНОЙ ОЦЕНКИ ВСЕХ МОДЕЛЕЙ ===\n")

    # ==========================================
    # ОЦЕНКА МОДЕЛЕЙ PYTORCH (Великая четверка)
    # ==========================================
    for model_name, (file_name, init_func) in models_config.items():
        weights_path = project_root / file_name

        if not weights_path.exists():
            print(f"[{model_name}] Пропуск: файл весов не найден.")
            continue

        print(f"[{model_name}] Расчет метрик...")
        model = init_func()

        if model_name == "SSD":
            in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        elif model_name == "Faster R-CNN":
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()

        metric = MeanAveragePrecision(class_metrics=False)

        # Переменные для ручного расчета Precision и Recall (Порог уверенности 0.5, IoU 0.5)
        tp, fp, fn = 0, 0, 0
        conf_threshold = 0.5
        iou_threshold = 0.5

        with torch.no_grad():
            for images, targets in data_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                preds = model(images)

                # Обновляем mAP
                metric.update(preds, targets)

                # Ручной расчет TP, FP, FN для каждой картинки
                for pred, target in zip(preds, targets):
                    pred_boxes = pred['boxes'][pred['scores'] > conf_threshold]
                    gt_boxes = target['boxes']

                    if len(pred_boxes) == 0:
                        fn += len(gt_boxes)
                        continue

                    if len(gt_boxes) == 0:
                        fp += len(pred_boxes)
                        continue

                    # Считаем матрицу пересечений (IoU)
                    ious = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()

                    for p_idx in range(len(pred_boxes)):
                        max_iou, max_idx = ious[p_idx].max(0)
                        if max_iou.item() > iou_threshold and max_idx.item() not in matched_gt:
                            tp += 1
                            matched_gt.add(max_idx.item())
                        else:
                            fp += 1
                    fn += len(gt_boxes) - len(matched_gt)

        # Итоговый расчет
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        map_results = metric.compute()
        map_50_95 = map_results['map'].item()
        map_50 = map_results['map_50'].item()

        # Добавляем в таблицу
        results_df.loc[len(results_df)] = [
            model_name, round(precision, 4), round(recall, 4), round(f1, 4),
            round(map_50_95, 4), round(map_50, 4), round(map_50_95, 4)
        ]

    # ==========================================
    # ОЦЕНКА МОДЕЛИ YOLO
    # ==========================================
    print(f"[YOLOv8] Расчет метрик...")
    yolo_path = project_root / 'runs/detect/train/weights/best.pt'  # ПРОВЕРЬ ЭТОТ ПУТЬ!
    yaml_path = project_root / "data" / "brain-tumor.yaml"

    if yolo_path.exists():
        model_yolo = YOLO(str(yolo_path))
        # Запускаем валидацию
        yolo_metrics = model_yolo.val(data=str(yaml_path), split='train', workers=0)

        y_prec = yolo_metrics.results_dict['metrics/precision(B)']
        y_rec = yolo_metrics.results_dict['metrics/recall(B)']
        y_map50 = yolo_metrics.results_dict['metrics/mAP50(B)']
        y_map5095 = yolo_metrics.results_dict['metrics/mAP50-95(B)']
        y_f1 = 2 * (y_prec * y_rec) / (y_prec + y_rec) if (y_prec + y_rec) > 0 else 0.0

        results_df.loc[len(results_df)] = [
            'YOLOv8', round(y_prec, 4), round(y_rec, 4), round(y_f1, 4),
            round(y_map5095, 4), round(y_map50, 4), round(y_map5095, 4)
        ]
    else:
        print("[YOLOv8] Пропуск: веса best.pt не найдены.")

    # ==========================================
    # СОХРАНЕНИЕ ТАБЛИЦЫ И ОТРИСОВКА
    # ==========================================
    print("\n=== СВОДНАЯ ТАБЛИЦА МЕТРИК ===")
    print(results_df.to_string(index=False))

    # Сохраняем в CSV (легко открыть в Excel)
    csv_path = project_root / "results" / "final_metrics_summary.csv"
    results_df.to_csv(csv_path, index=False, sep=';')
    print(f"\nТаблица сохранена в: {csv_path}")

    # Рисуем гистограмму
    plt.figure(figsize=(12, 6))

    x = np.arange(len(results_df['Model']))
    width = 0.2  # Ширина столбцов

    plt.bar(x - width * 1.5, results_df['Precision'], width, label='Precision', color='#4CAF50')
    plt.bar(x - width * 0.5, results_df['Recall'], width, label='Recall', color='#2196F3')
    plt.bar(x + width * 0.5, results_df['F1-score'], width, label='F1-score', color='#FFC107')
    plt.bar(x + width * 1.5, results_df['mAP@50'], width, label='mAP@50', color='#9C27B0')

    plt.title('Сравнительный анализ метрик качества моделей', fontsize=14, fontweight='bold')
    plt.ylabel('Значение метрики', fontsize=12)
    plt.xticks(x, results_df['Model'], fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Подписи значений над столбцами
    for i in range(len(results_df)):
        plt.text(x[i] - width * 1.5, results_df['Precision'][i] + 0.02, str(results_df['Precision'][i]), ha='center',
                 fontsize=8, rotation=90)
        plt.text(x[i] - width * 0.5, results_df['Recall'][i] + 0.02, str(results_df['Recall'][i]), ha='center',
                 fontsize=8, rotation=90)
        plt.text(x[i] + width * 0.5, results_df['F1-score'][i] + 0.02, str(results_df['F1-score'][i]), ha='center',
                 fontsize=8, rotation=90)
        plt.text(x[i] + width * 1.5, results_df['mAP@50'][i] + 0.02, str(results_df['mAP@50'][i]), ha='center',
                 fontsize=8, rotation=90)

    plot_path = project_root / "results" / "plots" / "final_metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Диаграмма успешно сохранена в: {plot_path}")


if __name__ == '__main__':
    evaluate_all_comprehensive()