import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("Чтение данных")
data = pd.read_parquet("../1_data/1_2_intermediate data/dataset.parquet")

print("Создание обучающей и целевой выборки")
X = data.iloc[:, :-1]
y = data["class"]

print("Создание скйелера")
std_scaler = StandardScaler()

X_scaled = std_scaler.fit_transform(X=X)

print("Разделение данных на обучающую и тестовую выборку")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print("Создание модели")
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")

print("Обучение модели")
svm_model.fit(X=X_train, y=y_train)

print("Предсказывания модели")
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Clf report: ")
print(classification_report(y_test, y_pred))
