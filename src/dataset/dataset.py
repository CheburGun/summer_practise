import os
import shutil
import zipfile
import urllib.request
from pathlib import Path


def clean_and_prepare_dataset():
    """
    Полностью очищает папку data/raw от старых дубликатов
    и делает одну чистую, идеальную распаковку датасета.
    """
    raw_dir = Path("data/raw")

    # Если папка существует, удаляем её вместе со всеми дубликатами
    if raw_dir.exists():
        print("Очистка папки data/raw от дубликатов...")
        shutil.rmtree(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
    zip_path = raw_dir / "brain-tumor.zip"

    print("Скачивание чистой копии медицинского датасета Brain Tumor...")
    urllib.request.urlretrieve(url, zip_path)

    print("Распаковка...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Распаковываем прямо в корень папки data/raw
        zip_ref.extractall(raw_dir)

    # Удаляем временный архив
    if zip_path.exists():
        os.remove(zip_path)

    print("Данные успешно подготовлены!")

    # Проверяем финальный результат
    all_images = list(raw_dir.rglob('*.jpg'))
    print(f"\n--- Итоговый отчет по данным ---")
    print(f"Всего уникальных снимков МРТ найдено: {len(all_images)}")

    if all_images:
        # Покажем точный путь к первой картинке, чтобы знать структуру
        print(f"Пример точного пути к файлу: {all_images[0]}")
    print("--------------------------------")


if __name__ == "__main__":
    clean_and_prepare_dataset()

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T


class MedicalDataset(Dataset):
    """
    Кастомный загрузчик данных для PyTorch.
    Читает разметку YOLO (.txt) и конвертирует ее в формат Faster R-CNN.
    """

    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms or T.ToTensor()
        # Собираем список всех изображений
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_c, y_c, width, height = map(float, line.strip().split())

                    # Математика перевода из YOLO в формат Pascal VOC [xmin, ymin, xmax, ymax]
                    xmin = (x_c - width / 2) * w
                    ymin = (y_c - height / 2) * h
                    xmax = (x_c + width / 2) * w
                    ymax = (y_c + height / 2) * h

                    boxes.append([xmin, ymin, xmax, ymax])
                    # Важно: Faster R-CNN резервирует 0 класс под "фон".
                    # Поэтому сдвигаем наши классы (negative/positive) на +1
                    labels.append(int(class_id) + 1)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        img = self.transforms(img)
        return img, target