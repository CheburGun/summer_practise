import os
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils as det_utils

# Импортируем правильный класс из файла dataset
from src.dataset.dataset import MedicalDataset


def train_ssd_model(epochs=3, batch_size=2):
    print("Начинаем подготовку модели SSDLite...")
    device = torch.device('cpu')

    # 1. Загрузка данных (точно как в твоем faster_rcnn.py)
    base_dir = Path(os.getcwd()) / "data" / "raw" / "images" / "train"
    labels_dir = Path(os.getcwd()) / "data" / "raw" / "labels" / "train"

    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    # Локальная функция обработки батчей
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True)

    # 2. Инициализация легковесной модели (SSDLite + MobileNetV3)
    num_classes = 3  # 0 (фон), 1 (negative), 2 (positive)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    # Меняем "голову" сети под наше количество классов (3 класса)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    model.to(device)

    # 3. Настройка оптимизатора
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    # 4. Цикл обучения
    print(f"Старт обучения на {epochs} эпох...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            print(f"  Эпоха {epoch + 1}/{epochs} | Батч {i + 1}/{len(data_loader)} | Loss: {losses.item():.4f}")

        print(f"--- Конец эпохи {epoch + 1} | Средний Loss: {epoch_loss / len(data_loader):.4f} ---")

    # 5. Сохранение результатов
    torch.save(model.state_dict(), 'ssd_weights.pth')
    print("Обучение завершено! Веса сохранены в 'ssd_weights.pth'")