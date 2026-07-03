import os
from pathlib import Path
from ultralytics import YOLO


def create_yolo_yaml():
    """
    Создает конфигурационный файл для YOLO с абсолютными путями
    строго под твой компьютер, чтобы не было ошибок 'File not found'.
    """
    # Получаем абсолютный путь к папке data/raw
    base_path = Path(os.getcwd()) / "data" / "raw"
    yaml_path = base_path / "custom_dataset.yaml"

    # YOLO требует указать корень, и где лежат картинки
    yaml_content = f"""
path: {base_path}
train: images/train
val: images/val

names:
  0: negative
  1: positive
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    return str(yaml_path)


def train_yolo_baseline(epochs=10, batch_size=4, imgsz=640):
    """Обучает базовую модель YOLOv8n."""
    print("1. Подготовка конфигурации данных...")
    data_yaml = create_yolo_yaml()

    print("2. Загрузка архитектуры YOLOv8n...")
    # Используем 'yolov8n.pt', который уже скачался в корень проекта
    model = YOLO("yolov8n.pt")

    print("3. Старт обучения...")
    # Метод train сам всё разобьет на батчи и посчитает лосс/метрики
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project="results",  # Папка, куда по заданию сохраняем итоги
        name="yolo_baseline",  # Название папки с этим экспериментом
        plots=True  # Сразу строим графики (loss, mAP), нужные для отчета
    )

    print("Обучение завершено! Все графики и веса лежат в results/yolo_baseline")
    return model