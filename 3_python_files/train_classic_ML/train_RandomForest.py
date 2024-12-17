import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Чтение данных")
data = pd.read_parquet("../1_data/1_2_intermediate data/dataset.parquet")

print("Создание обучающей и целевой выборки")
X = data.iloc[:, :-1]
y = data["class"]

print("Разделение данных на обучающую и тестовую выборку")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Создание модели")
rfc_model = RandomForestClassifier(n_estimators=50, random_state=42)

print("Обучение модели")
rfc_model.fit(X=X_train, y=y_train)

print("Предсказывания модели")
y_pred = rfc_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Clf report: ")
print(classification_report(y_test, y_pred))
