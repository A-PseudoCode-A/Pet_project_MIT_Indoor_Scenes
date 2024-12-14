import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

img = cv2.imread(
    "../1_data/1_1_raw_data/rooms_images/airport_inside/airport_inside_0001.jpg"
)
print(img)
img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap="gray")


img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
plt.imshow(img, cmap="gray")


# Укажите путь к директории
root_dir = r"C:\All_Enough\Pseudocode\Projects\Pet_project_clf_rooms_mit\1_data\1_1_raw_data\rooms_images"

# Подсчитываем количество папок
folder_count = sum(len(subdirs) for _, subdirs, _ in os.walk(root_dir))

print(f"Количество вложенных папок: {folder_count}")

folders_list = [
    folder
    for folder in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, folder))
]
folders_dict = {}
for class_number, class_name in enumerate(folders_list):
    folders_dict[class_name] = class_number


print(folders_dict)
# Укажите путь к директории

root_dir = r"C:\All_Enough\Pseudocode\Projects\Pet_project_clf_rooms_mit\1_data\1_1_raw_data\rooms_images"

# Создаем список для всех путей к изображениям
image_paths = []

# Проходим по всем подпапкам
for current_path, subdirs, files in os.walk(root_dir):
    for file in files:
        # Проверяем, является ли файл изображением
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
            # Добавляем полный путь к изображению
            image_paths.append(os.path.join(current_path, file))

# Вывод списка изображений
print(f"Найдено {len(image_paths)} изображений")
for path in image_paths[600:620]:  # Выводим первые 5 изображений для примера
    print(path)

print(image_paths[1856])

cnt = 0
images = []
for img in image_paths:
    img = cv2.imread(filename=img)
    cnt += 1
    print(cnt)

    if img is None:
        continue
    else:
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        images.append(img)


print("Картины хранятся в images")
print(f"Размер каждый картинки: {len(images[1000])} x {len(images[1000][0])}")

columns_name = [f"column_{i}" for i in range(1, 257)]

images = np.array(images)
df = pd.DataFrame(images.reshape(15590, -1))
print(df.shape)  # (15590, 65536)

df.to_parquet("lol.parquet")
