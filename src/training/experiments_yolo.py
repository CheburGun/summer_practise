import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path


def run_yolo_experiments():
    project_root = Path(__file__).resolve().parents[2]
    yaml_path = project_root / "data" / "raw"/ "brain-tumor.yaml"

    experiments = {
        "Baseline": {"epochs": 10, "batch": 4, "lr0": 0.01, "optimizer": "auto"},
        "High LR": {"epochs": 10, "batch": 4, "lr0": 0.1, "optimizer": "auto"},
        "Large Batch": {"epochs": 10, "batch": 8, "lr0": 0.01, "optimizer": "auto"},
        "AdamW Opt": {"epochs": 10, "batch": 4, "lr0": 0.001, "optimizer": "AdamW"}
    }

    results_map = {}

    print("=== СТАРТ ЭКСПЕРИМЕНТОВ ДЛЯ YOLO ===\n")

    for exp_name, config in experiments.items():
        print(f"\n--- Запуск {exp_name} ---")

        model = YOLO('yolov8n.pt')

        # Функция train возвращает объект с метриками
        metrics = model.train(
            data=str(yaml_path),
            epochs=config["epochs"],
            batch=config["batch"],
            lr0=config["lr0"],
            optimizer=config["optimizer"],
            name=f"YOLO_{exp_name.replace(' ', '_')}",
            workers=0
        )

        # Извлекаем общий mAP50-95
        final_map = metrics.box.map
        print(f"✅ {exp_name} завершен! mAP: {final_map:.4f}")

        results_map[exp_name] = final_map

    # ==========================================
    # БЛОК ОТРИСОВКИ СВОДНОГО ГРАФИКА
    # ==========================================
    print("\nГенерация сводного графика...")

    names = list(results_map.keys())
    values = list(results_map.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#8E44AD', '#2980B9', '#27AE60', '#E67E22'])

    plt.title('YOLOv8: Влияние гиперпараметров на точность (mAP)', fontsize=14, fontweight='bold')
    plt.ylabel('mAP (Mean Average Precision)', fontsize=12)
    plt.ylim(0, max(values) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom',
                 fontweight='bold')

    save_path = project_root / "results" / "plots" / "yolo_hyperparams_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График успешно сохранен: {save_path}")


if __name__ == '__main__':
    run_yolo_experiments()