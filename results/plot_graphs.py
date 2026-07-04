import matplotlib.pyplot as plt

# Эпохи (ось X) - у всех 4 моделей строго 5 эпох
epochs = [1, 2, 3, 4, 5]

# ВНИМАНИЕ: Замени эти числа на реальные значения Среднего Loss из терминала!
loss_faster_rcnn = [0.1823, 0.1216, 0.1029, 0.0889, 0.0797]
loss_ssd = [5.1317, 3.4339, 2.9782, 2.5982, 2.2618]
loss_retinanet = [1.4582, 1.5796, 1.5805, 1.5438, 1.5562]
loss_fcos        = [1.4230, 1.1295, 1.0634, 1.0377, 0.9884]

# Создаем большое полотно для графика
plt.figure(figsize=(10, 6))

# Рисуем линии для каждой модели (с разными цветами и маркерами для красоты)
plt.plot(epochs, loss_faster_rcnn, marker='^', linestyle='-', color='g', label='Faster R-CNN (ResNet50)', linewidth=2)
plt.plot(epochs, loss_ssd, marker='o', linestyle='-', color='b', label='SSD (MobileNetV3)', linewidth=2)
plt.plot(epochs, loss_retinanet, marker='d', linestyle='-', color='m', label='RetinaNet (ResNet50)', linewidth=2)
plt.plot(epochs, loss_fcos, marker='s', linestyle='--', color='r', label='FCOS (ResNet50)', linewidth=2)

# Оформление по академическим стандартам
plt.title('Сравнение функции потерь (Loss) моделей на базе PyTorch', fontsize=14, fontweight='bold')
plt.xlabel('Эпоха обучения', fontsize=12)
plt.ylabel('Среднее значение Loss', fontsize=12)

# Настройка осей и сетки
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

# Сохраняем картинку в высоком качестве (подойдет для вставки в Word)
plt.savefig('pytorch_models_comparison.png', dpi=300, bbox_inches='tight')
print("График успешно сохранен в файл 'pytorch_models_comparison.png'")

# Показываем график на экране
plt.show()