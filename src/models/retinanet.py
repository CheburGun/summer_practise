import os
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader

# Импортируем наш универсальный датасет
from src.dataset.dataset import MedicalDataset


def train_retinanet_model(epochs=5, batch_size=2):
    print("Начинаем подготовку модели RetinaNet...")
    device = torch.device('cpu')

    # 1. Загрузка данных
    base_dir = Path(os.getcwd()) / "data" / "raw" / "images" / "train"
    labels_dir = Path(os.getcwd()) / "data" / "raw" / "labels" / "train"

    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    def collate_fn(batch):
        return tuple(zip(*batch))

    # Обязательно drop_last=True
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0, drop_last=True)

    # 2. Инициализация модели RetinaNet
    num_classes = 3  # 0 (фон), 1 (negative), 2 (positive)

    # PyTorch сам соберет правильную "голову" под 3 класса
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights_backbone="DEFAULT",
        num_classes=num_classes
    )

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
    torch.save(model.state_dict(), 'retinanet_weights.pth')
    print("Обучение завершено! Веса сохранены в 'retinanet_weights.pth'")