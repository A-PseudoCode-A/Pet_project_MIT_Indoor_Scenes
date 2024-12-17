import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def graphics_det(title, x_name, y_name, rotation=None):
    plt.figure(figsize=(15, 11))
    plt.xlabel(xlabel=x_name)
    plt.ylabel(ylabel=y_name)
    plt.title(label=title)
    if rotation is not None:
        plt.xticks(rotation=rotation)

plt.rcParams.update({'font.size': 10})
data = pd.read_parquet("../1_data/1_3_ready_data/dataset.parquet")

cnt_images_data = data.iloc[:, [0, -1]].groupby(by='class', axis=0).count().iloc[:, 0].sum()

graphics_det(
    title="Распределение изображений в классах",
    x_name="Классы",
    y_name="Количество",
    rotation=60,
)
sns.countplot(data=data, x="class")
