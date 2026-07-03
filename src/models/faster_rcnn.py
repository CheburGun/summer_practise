import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from src.dataset.dataset import MedicalDataset
from pathlib import Path
import os


def get_faster_rcnn_model(num_classes):
    # Загружаем предобученную базу ResNet50
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменяем классификатор под наше количество классов
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_faster_rcnn_model(epochs=5, batch_size=2):
    print("1. Подготовка данных для PyTorch...")
    base_dir = Path(os.getcwd()) / "data" / "raw" / "images" / "train"
    labels_dir = Path(os.getcwd()) / "data" / "raw" / "labels" / "train"

    dataset = MedicalDataset(images_dir=str(base_dir), labels_dir=str(labels_dir))

    # Специфичный обработчик батчей для детекции
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"2. Инициализация модели Faster R-CNN на устройстве: {device}")

    # У нас 3 класса: 0 (фон), 1 (negative), 2 (positive)
    model = get_faster_rcnn_model(num_classes=3)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    print("3. Старт обучения...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Faster R-CNN сам считает все 4 вида loss-функций в режиме train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # Выводим лог каждые 50 шагов, чтобы видеть процесс
            print(f"  Эпоха {epoch + 1}/{epochs} | Батч {i}/{len(data_loader)} | Loss: {losses.item():.4f}")

        print(f"=== Итог эпохи {epoch + 1}: Средняя ошибка = {epoch_loss / len(data_loader):.4f} ===")

    print("Обучение Faster R-CNN завершено!")
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/faster_rcnn_weights.pth")
    return model