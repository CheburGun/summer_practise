import os
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset.dataset import MedicalDataset


def run_ssd_experiments():
    device = torch.device('cpu')
    project_root = Path(__file__).resolve().parents[2]

    base_dir = project_root / "data" / "raw" / "images" / "train"
    labels_dir = project_root / "data" / "raw" / "labels" / "train"
    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    def collate_fn(batch):
        return tuple(zip(*batch))

    experiments = {
        "Baseline": {"lr": 0.0001, "batch_size": 2, "optimizer": "Adam"},
        "High LR": {"lr": 0.001, "batch_size": 2, "optimizer": "Adam"},
        "Large Batch": {"lr": 0.0001, "batch_size": 4, "optimizer": "Adam"},
        "SGD Opt": {"lr": 0.001, "batch_size": 2, "optimizer": "SGD"}
    }

    # Словарь для сбора финальных метрик
    results_map = {}

    print("=== СТАРТ ЭКСПЕРИМЕНТОВ ДЛЯ SSD ===\n")

    for exp_name, config in experiments.items():
        print(f"\n--- Запуск {exp_name} ---")

        data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                 collate_fn=collate_fn, num_workers=0, drop_last=True)

        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, 3)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        if config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(params, lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=0.9)

        model.train()
        for epoch in range(5):
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

        model.eval()
        metric = MeanAveragePrecision(class_metrics=False)
        eval_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        with torch.no_grad():
            for images, targets in eval_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                preds = model(images)
                metric.update(preds, targets)

        final_map = metric.compute()['map'].item()
        print(f"✅ Итог {exp_name} -> mAP: {final_map:.4f}")

        # Сохраняем результат для графика
        results_map[exp_name] = final_map

    # ==========================================
    # БЛОК ОТРИСОВКИ СВОДНОГО ГРАФИКА
    # ==========================================
    print("\nГенерация сводного графика...")

    names = list(results_map.keys())
    values = list(results_map.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#4CAF50', '#2196F3', '#FFC107', '#F44336'])

    plt.title('SSD: Влияние гиперпараметров на точность (mAP)', fontsize=14, fontweight='bold')
    plt.ylabel('mAP (Mean Average Precision)', fontsize=12)
    plt.ylim(0, max(values) + 0.1)  # Динамическая высота оси Y
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем точные цифры над каждым столбцом
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom',
                 fontweight='bold')

    # Сохраняем в папку plots
    save_path = project_root / "results" / "plots" / "ssd_hyperparams_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График успешно сохранен: {save_path}")


if __name__ == '__main__':
    run_ssd_experiments()