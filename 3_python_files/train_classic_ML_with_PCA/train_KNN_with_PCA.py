import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Чтение данных")
data = pd.read_parquet("../1_data/1_2_ready_data/dataset.parquet")
print("   Успешно 🆗")

print("\nВыделение целевого признака")
X = data.iloc[:, :-1]
y = data["class"]
print("   Успешно 🆗")

print("\nСтандартизация данных")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   Успешно 🆗")

print("\nСнижение размерности")
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)
print("   Успешно 🆗")

print("\nРазделение данных на обучающую и тестовую выборку")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print("   Успешно 🆗")

print("\nСоздание модели")
rfc_model = KNeighborsClassifier()
print("   Успешно 🆗")

print("\nОбучение модели")
rfc_model.fit(X=X_train, y=y_train)
print("   Успешно 🆗")

print("\nПредсказания модели")
y_pred = rfc_model.predict(X_test)
print("   Успешно 🆗")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Clf report: ")
print(classification_report(y_test, y_pred))
