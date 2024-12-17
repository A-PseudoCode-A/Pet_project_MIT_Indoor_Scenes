import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


class VGG(nn.Module):
    def __init__(self, name, num_classes=67, dropout=0.5):
        super().__init__()
        self.cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

        self.features = self.make_layers(self.cfg)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.flatten(x)
        out = self.classifier(x)
        return out

    def make_layers(self, cfg):
        layres = []
        in_channels = 3
        for value in cfg:
            if value == "M":
                layres += [nn.MaxPool2d(2)]
            else:
                conv2d = nn.Conv2d(in_channels, value, (3, 3), padding=1)
                layres += [conv2d, nn.ReLU(True)]
                in_channels = value
        return nn.Sequential(*layres)


# Кастомный Dataset для наших данных
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Загрузка данных
print("Чтение данных")
data = torch.load("../../1_data/1_3_ready_data/final_dataset.pt")
images = data["images"]
labels = data["labels"]
print("   Успешно 🆗")

# Создание CustomDataset
print("\nСоздание CustomDataset")
dataset = CustomDataset(images, labels)
print("   Успешно 🆗")

# Разделение данных на обучающую, валидационную и тестовую выборки

print("\nРазделение данных на обучающую, валидационную и тестовую выборки")
train_data, test_data = random_split(dataset, [0.9, 0.1])
train_data, val_data = random_split(train_data, [0.8, 0.2])
print("   Успешно 🆗")

# Создание DataLoader для обучения нейронной сети с помощью батчей
print("\nСоздаие DataLoader")
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
vak_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
print("   Успешно 🆗")

# Создание объекта класса VGG
print("\nСоздание модели")
model = VGG(name="vgg_11").to("cuda")
print("   Успешно 🆗")

# Создание функции потерь и оптимизатора градиентного бустинга
print("\nСоздание функции потерь и оптимизатора градиентного бустинга")
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("   Успешно 🆗")

# Создание вспомогательных списков, которые будут предоставлять дополнительную информацию по время обучения
print("\nСоздание вспомогательных переменн")
EPOCHS = 50
train_loss = []
train_acc = []
val_loss = []
val_acc = []
print("   Успешно 🆗")

# Цикл обучения
print("\nНачало обучения модели")
for epoch in range(EPOCHS):
    true_answer = 0
    running_train_loss = []

    model.train()
    train_loop = tqdm(train_loader, leave=False)

    for x, targets in train_loop:
        x, targets = x.to("cuda"), targets.to("cuda")

        # Прямой проход
        pred = model(x)
        loss = loss_function(pred, targets)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}"
        )

    running_train_acc = true_answer / len(train_data)
    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    # Валидация
    model.eval()
    with torch.no_grad():
        running_val_loss = []
        true_answer = 0

        for x, targets in vak_loader:
            x, targets = x.to("cuda"), targets.to("cuda")

            # Прямой проход
            pred = model(x)
            loss = loss_function(pred, targets)

            pred = model(x)
            loss = loss_function(pred, targets)

            running_val_loss.append(loss.item())
            true_answer += (pred.argmax(dim=1) == targets).sum().item()

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)
        running_val_acc = true_answer / len(val_data)

        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        print(
            f"Epoch [{epoch+1} / {EPOCHS}], train_loss = {mean_train_loss:.4f}, train_acc = {running_train_acc:.4f}, val_loss = {mean_val_loss:.4f}, val_acc = {running_val_acc:.4f}"
        )
