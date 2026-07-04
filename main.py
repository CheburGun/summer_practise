from src.models.yolo import train_yolo_baseline
from src.models.faster_rcnn import train_faster_rcnn_model
from src.models.fcos import train_fcos_model

if __name__ == "__main__":
    # Выключаем YOLO, так как она уже обучена
    # print("=== Старт ML Pipeline: Базовая модель (YOLO) ===")
    # train_yolo_baseline(epochs=10, batch_size=4, imgsz=640)

    # print("\n=== Старт ML Pipeline: Вторая модель (Faster R-CNN) ===")
    # # Ставим 5 эпох и размер батча 2 (Faster R-CNN гораздо тяжелее для процессора)
    # train_faster_rcnn_model(epochs=5, batch_size=2)
    train_fcos_model(epochs=5, batch_size=2)