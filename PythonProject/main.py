# Инструкция по запуску: загрузить приложенный файл data.csv в браузерный jupiter. Скопировать в блокнот этот код. выбрать ядро Pyodide
# Программа будет работать несколько минут

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

# Считаем правильно данные
df = pd.read_csv("data/data.csv", header = None)

# Посмотрим на данные:
df.head()
df.info()
df.describe()

features = df.iloc[:,1:-1]
target = df.iloc[:,-1]

features_train, features_test, target_train, target_test = train_test_split(
    features, target,
    test_size=0.5,
    random_state=42,
    stratify=target
)

# Алгоритмы KNN и SVM требуют нормализации функций.
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Алгоритм KNN
knn = KNeighborsClassifier()

param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring="accuracy")
grid_knn.fit(features_train_scaled, target_train)
print ("--------------------------------")
print("KNN оптимальные параметры:", grid_knn.best_params_)
print("KNN CV accuracy:", grid_knn.best_score_)
print ("--------------------------------")

# Алгоритм RandomForest
rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="accuracy")
grid_rf.fit(features_train, target_train)
print("RF оптимальные параметры:", grid_rf.best_params_)
print("RF CV accuracy:", grid_rf.best_score_)
print ("--------------------------------")

# SVM с RBF ядром

svm = SVC(kernel="rbf")

param_grid_svm = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"]
}

grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring="accuracy")
grid_svm.fit(features_train_scaled, target_train)

print("SVM оптимальные параметры:", grid_svm.best_params_)
print("SVM CV accuracy:", grid_svm.best_score_)
print ("--------------------------------")

# лучший KNN
knn_best = grid_knn.best_estimator_
target_pred_knn = knn_best.predict(features_test_scaled)
print("KNN test accuracy:", accuracy_score(target_test, target_pred_knn))
print(classification_report(target_test, target_pred_knn))
print ("--------------------------------")

# лучший RF
rf_best = grid_rf.best_estimator_
target_pred_rf = rf_best.predict(features_test)
print("RF test accuracy:", accuracy_score(target_test, target_pred_rf))
print(classification_report(target_test, target_pred_rf))
print ("--------------------------------")

# лучший SVM
svm_best = grid_svm.best_estimator_
target_pred_svm = svm_best.predict(features_test_scaled)
print("SVM test accuracy:", accuracy_score(target_test, target_pred_svm))
print(classification_report(target_test, target_pred_svm))
print ("--------------------------------")

# Судя по результатам понимаем, что лучше всего здесь подходит классификатор RF.  SVM с RBF ядром и
# KNN показывают меньшую точность. Сравним эти классификаторы на очищенных данных.

# Теперь очистим данные и проделаем то же самое. Сначала удалим шумовые данные, затем неинформативные признаки.
# ----------------- последовательность: шумовые -> неинформативные -----------------

# Для удаления шумовых данных используем IsolationForest

start = datetime.now()

iso = IsolationForest(contamination=0.05, random_state=42)  # 5% выбросов
outliers = iso.fit_predict(features_train)
features_train_clean = features_train[outliers == 1]
target_train_clean = target_train[outliers == 1]

# Теперь удалим неинформативные признаки, используем VarianceThreshold
selector = VarianceThreshold(threshold=1e-5)
features_train_clean = selector.fit_transform(features_train_clean)
features_test_clean = selector.transform(features_test)
assert features_train_clean.shape[1] == features_test_clean.shape[1], \
    f"Ошибка: количество признаков не совпадает! Train: {features_train_clean.shape[1]}, Test: {features_test_clean.shape[1]}"

# Масштабирование
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_clean)
features_test_scaled = scaler.transform(features_test_clean)

# KNN
param_grid_knn = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(features_train_scaled, target_train_clean)
knn_best = grid_knn.best_estimator_
target_pred_knn = knn_best.predict(features_test_scaled)

# RandomForest
param_grid_rf = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(features_train_clean, target_train_clean)
rf_best = grid_rf.best_estimator_
target_pred_rf = rf_best.predict(features_test_clean)

# SVM
param_grid_svm = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
grid_svm = GridSearchCV(SVC(kernel='rbf'), param_grid_svm, cv=5, scoring='accuracy')
grid_svm.fit(features_train_scaled, target_train_clean)
svm_best = grid_svm.best_estimator_
target_pred_svm = svm_best.predict(features_test_scaled)

print("шумные данные -> неинформативные признаки")
print("KNN accuracy:", accuracy_score(target_test, target_pred_knn))
print(classification_report(target_test, target_pred_knn))
print("--------------------------------")
print("RF accuracy:", accuracy_score(target_test, target_pred_rf))
print(classification_report(target_test, target_pred_rf))
print("--------------------------------")
print("SVM accuracy:", accuracy_score(target_test, target_pred_svm))
print(classification_report(target_test, target_pred_svm))
end = datetime.now()
print(f"Время выполнения: {end - start}")
print("--------------------------------")

# ----------------- последовательность: неинформативные -> шумовые -----------------

# Сначала удаляем неинформативные признаки
start = datetime.now()

selector = VarianceThreshold(threshold=1e-5)
features_train_clean2 = selector.fit_transform(features_train)
features_test_clean2 = selector.transform(features_test)

# Потом удаляем шумовые данные
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(features_train_clean2)
features_train_clean2 = features_train_clean2[outliers == 1]
target_train_clean2 = target_train[outliers == 1]
assert features_train_clean2.shape[1] == features_test_clean2.shape[1], \
    f"Ошибка: количество признаков не совпадает! Train: {features_train_clean2.shape[1]}, Test: {features_test_clean2.shape[1]}"

# Масштабирование
scaler = StandardScaler()
features_train_scaled2 = scaler.fit_transform(features_train_clean2)
features_test_scaled2 = scaler.transform(features_test_clean2)

# KNN
param_grid_knn = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(features_train_scaled2, target_train_clean2)
knn_best = grid_knn.best_estimator_
target_pred_knn = knn_best.predict(features_test_scaled2)

# RandomForest
param_grid_rf = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(features_train_clean2, target_train_clean2)
rf_best = grid_rf.best_estimator_
target_pred_rf = rf_best.predict(features_test_clean2)

# SVM
param_grid_svm = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
grid_svm = GridSearchCV(SVC(kernel='rbf'), param_grid_svm, cv=5, scoring='accuracy')
grid_svm.fit(features_train_scaled2, target_train_clean2)
svm_best = grid_svm.best_estimator_
target_pred_svm = svm_best.predict(features_test_scaled2)

print("неинформативные признаки -> шумные данные")
print("KNN accuracy:", accuracy_score(target_test, target_pred_knn))
print(classification_report(target_test, target_pred_knn))
print("--------------------------------")
print("RF accuracy:", accuracy_score(target_test, target_pred_rf))
print(classification_report(target_test, target_pred_rf))
print("--------------------------------")
print("SVM accuracy:", accuracy_score(target_test, target_pred_svm))
print(classification_report(target_test, target_pred_svm))
end = datetime.now()
print(f"Время выполнения: {end - start}")
print("--------------------------------")

# Общий вывод: на этом датасете очистка данных не дает существенного прироста в точности для всех моделей. Порядок удаления "грязных данных"
# здесь тоже не дает особой разницы. Во всех случаях наилучшим классификатором является RF.
# Замечание: идейно время выполнения немного меньше, когда сначала удаляются неинформативные признаки, так как меньше признаков для расчёта.