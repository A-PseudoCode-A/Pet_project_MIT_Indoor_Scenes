import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#! Изменить путь для создания файла parquet


class CreateDatasetForClassicML:
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
        self.images_list = []

        for current_path, subdirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
                ):
                    image_path = os.path.join(current_path, file)
                    class_label = self.folders_dict[os.path.basename(current_path)]
                    self.images_list.append((image_path, class_label))

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
        print("    Создание списка классов изображений:")
        self.create_a_list_of_classes()
        print("    Создание списка всех изображений в виде numpy.array:")
        self.create_image_list()
        print("    Создание датасета")
        self.create_dataset()


class CreateDatasetForNeuralNetworks:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_transform(self):
        # Предобработка изображений
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Изменяем размер
                transforms.ToTensor(),  # Преобразуем в тензор
                transforms.Normalize(
                    [0.485, 0.456, 0.406],  # Нормализация (пример: ImageNet)
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def create_butch_files(self):
        # Загрузка данных
        dataset = datasets.ImageFolder(self.dataset_path, transform=self.transform)
        # Уменьшаем размер батча
        batch_size = 256
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Сохраняем данные частями
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Переносим данные на GPU (если доступно)
            images, labels = images.to(self.device), labels.to(self.device)

            # Сохраняем текущий батч
            torch.save(
                {"images": images, "labels": labels},
                f"../1_data/1_2_intermediate_data/processed_batch_{batch_idx}.pt",
            )

            # Очистка кэша
            torch.cuda.empty_cache()

        print("Processing complete!")

    def batch_file_counting(self):
        folder_path = Path("../1_data/1_2_intermediate_data")
        # Подсчитываем количество файлов
        file_count = sum(1 for file in folder_path.iterdir() if file.is_file())
        # Количество промежутоных файлов
        self.batch_files = [
            f"../1_data/1_2_intermediate_data/processed_batch_{i}.pt"
            for i in range(file_count)
        ]

    def create_dataset(self):
        # Функция для загрузки батча
        def load_batch(file):
            batch = torch.load(file)
            images = batch["images"].to("cpu")
            labels = batch["labels"].to("cpu")
            torch.cuda.empty_cache()  # Очистка кэша GPU
            return {"images": images, "labels": labels}

        # Используем многопоточность для загрузки всех файлов
        with ThreadPoolExecutor() as executor:
            batches = list(executor.map(load_batch, self.batch_files))

        # Объединяем данные
        all_images = torch.cat([batch["images"] for batch in batches])
        all_labels = torch.cat([batch["labels"] for batch in batches])

        # Сохраняем объединённые данные в один файл
        torch.save(
            {"images": all_images, "labels": all_labels},
            "../1_data/1_3_ready_data/final_dataset.pt",
        )
        print("Данные успешно объединены и сохранены в 'final_dataset.pt'.")

    def delete_butch_files(self):
        for file_path in self.batch_files:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                print(f"Файл {file_path} удалён.")
            else:
                print(f"Файл {file_path} не существует.")

    def __call__(self):
        self.create_transform()
        self.create_butch_files()
        self.create_dataset()
        self.delete_butch_files()


cls = CreateDatasetForClassicML(
    dataset_path=r"C:\All_Enough\Pseudocode\Projects\Pet_project_clf_rooms_mit\1_data\1_1_raw_data\rooms_images"
)
cls()

cls_2 = CreateDatasetForNeuralNetworks(
    dataset_path=r"C:\All_Enough\Pseudocode\Projects\Pet_project_clf_rooms_mit\1_data\1_1_raw_data\rooms_images"
)
cls_2()
