import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


#! Изменить путь для создания файла parquet


class CreateDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def create_a_list_of_classes(self):
        folders_list = [
            folder
            for folder in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, folder))
        ]

        self.folders_dict = {}
        for class_number, class_name in enumerate(folders_list):
            self.folders_dict[class_name] = class_number

        print(f"Найдено {len(self.folders_dict)} классов")
        print(f"Cписок всех классов: {self.folders_dict}")

    def create_image_list(self):
        self.images_list = []  # Список для хранения путей и классов изображений

        for current_path, subdirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
                ):
                    image_path = os.path.join(current_path, file)
                    class_label = self.folders_dict[
                        os.path.basename(current_path)
                    ]  # Класс текущего изображения
                    self.images_list.append(
                        (image_path, class_label)
                    )  # Добавляем путь и класс в виде кортежа

        # Вывод количества найденных изображений
        print(f"Найдено {len(self.images_list)} изображений")

    def create_dataset(self):
        self.dataset_images = []
        self.dataset_classes = []

        for image, num_class in self.images_list:
            image = cv2.imread(filename=image)
            if image is not None:
                image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
                image = cv2.resize(
                    src=image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC
                )
                self.dataset_images.append(image)
                self.dataset_classes.append(num_class)

        print(
            f"Размер каждой картинки: {len(self.dataset_images[1000])} x {len(self.dataset_images[1000][0])}"
        )

        self.dataset_images = np.array(self.dataset_images)
        self.dataset_classes = np.array(self.dataset_classes)

        self.dataset = pd.DataFrame(self.dataset_images.reshape(15590, -1))
        self.dataset["class"] = self.dataset_classes

        print(f"Размер датасета: {self.dataset.shape}")
        print("Конвертация датасета в файл parquet")

        self.dataset.to_parquet("dataset.parquet")

        print("Завершено успешно!")

    def __call__(self):
        print("    Выполнение всех команд для создания датасета:")
        print("    Создание списка классов изображений:")
        self.create_a_list_of_classes()
        print("    Создание списка всех изображений в виде numpy.array:")
        self.create_image_list()
        print("    Создание датасета")
        self.create_dataset()


cls = CreateDataset(
    dataset_path=r"C:\All_Enough\Pseudocode\Projects\Pet_project_clf_rooms_mit\1_data\1_1_raw_data\rooms_images"
)
cls()

data = pd.read_parquet("dataset.parquet")
