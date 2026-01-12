# Как запускать: положить в браузерный Jupiter файлы Test.csv, Train.csv, скопировать код в notebook, выбрать ядро Pyodide, запустить, (около 10 минут работает).
# По окончанию работы появится выходной файл Radustov_predictions.csv
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# 1. Загрузка данных
# =========================

# Знаем что классы сбалансированы, пропусков и Nan нет
train = pd.read_csv("Train.csv")
features = train.iloc[:, :-1].astype(float)
target = (train.iloc[:, -1] == 1).astype(int)

test = pd.read_csv("Test.csv", header=None).astype(float)
test.columns = features.columns

# =========================
# 2. Stratified CV
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
"""
# =========================
# 3. Остальные модели (кроме GradientBoosting)
# =========================
models = [
    ("PCA + LogisticRegression",
     Pipeline([("scaler", StandardScaler()), ("pca", PCA()),
               ("clf", LogisticRegression(solver="liblinear", max_iter=5000))]),
     {"pca__n_components": [50, 100, 200], "clf__C": [0.1, 1.0, 10.0]}),

    ("SelectKBest + LogisticRegression",
     Pipeline([("scaler", StandardScaler()), ("kbest", SelectKBest(f_classif)),
               ("clf", LogisticRegression(solver="liblinear", max_iter=5000))]),
     {"kbest__k": [50, 100, 200], "clf__C": [0.1, 1.0, 10.0]}),

    ("L1 LogisticRegression",
     Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(l1_ratio=0, solver="saga", max_iter=5000, random_state=42))]),
     {"clf__C": [0.1, 1.0, 10.0]}),

    ("RandomForest + SelectKBest",
     Pipeline([("kbest", SelectKBest(f_classif)),
               ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
     {"kbest__k": [100, 200, 300], "clf__n_estimators": [300, 500],
      "clf__max_depth": [10, 20], "clf__min_samples_split": [2, 5]}),

    ("XGBoost + SelectKBest",
     Pipeline([("kbest", SelectKBest(f_classif)),
               ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1))]),
     {"kbest__k": [100, 200, 300], "clf__n_estimators": [300, 500],
      "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5],
      "clf__subsample": [0.8, 1.0], "clf__colsample_bytree": [0.8, 1.0]})
]

results = []
grids_dict = {}

for name, pipe, param_grid in models:
    print(f"Обучаем {name} ...")
    grid = GridSearchCV(pipe, param_grid, scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"},
                        refit="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(features, target)
    results.append({"Model": name,
                    "ROC-AUC": grid.best_score_,
                    "Accuracy": grid.cv_results_["mean_test_accuracy"][grid.best_index_]})
    grids_dict[name] = grid

# =========================
# 4. Итоговая таблица
# =========================
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
print("\n=== Итоговые результаты ===")
print(results_df)
# посмотрим на результаты, отдельно сделаем градиентный бустинг
"""
# =========================
# 5. GradientBoosting + SelectKBest отдельно
# =========================
pipe_gb = Pipeline([("kbest", SelectKBest(f_classif)),
                    ("clf", GradientBoostingClassifier(random_state=42))])
param_grid_gb = {"kbest__k": [100, 200, 300],
                 "clf__n_estimators": [200, 400],
                 "clf__learning_rate": [0.05, 0.1],
                 "clf__max_depth": [3, 5],
                 "clf__subsample": [0.8, 1.0]}

grid_gb = GridSearchCV(
    pipe_gb,
    param_grid_gb,
    scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"},
    refit="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid_gb.fit(features, target)

best_index = grid_gb.best_index_
best_accuracy = grid_gb.cv_results_["mean_test_accuracy"][best_index]

print("\n=== Метрики GradientBoosting + SelectKBest ===")
print(f"Лучшие гиперпараметры: {grid_gb.best_params_}")
print(f"Лучший ROC-AUC (кросс-валидация): {grid_gb.best_score_:.4f}")
print(f"Accuracy (кросс-валидация): {best_accuracy:.4f}")

# лучшие метрики показал градиентный бустинг. Применим эту модель к тестовой выборке

final_model = grid_gb.best_estimator_

# =========================
# 6. Предсказания на тестовой выборке
# =========================
pred_classes = final_model.predict(test)
pred_proba = final_model.predict_proba(test)
confidence = np.max(pred_proba, axis=1)

result_df = pd.DataFrame({"class": pred_classes, "confidence": confidence})
output_path = "Radustov_predictions.csv"
result_df.to_csv(output_path, index=False)

print(f"\nФайл с предсказаниями сохранён: {output_path}")
print("Распределение предсказанных классов:")
print(result_df["class"].value_counts())
print(f"Средняя уверенность: {confidence.mean():.4f}")

print("Распределение предсказанных классов:")
print(result_df["class"].value_counts())
# видим, что баланс сохраняется
print(f"Средняя уверенность: {confidence.mean():.4f}")
