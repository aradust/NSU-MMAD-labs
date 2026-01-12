# Инструкция по запуску в браузерном jupiter: Создать Notebook, Выбрать ядро Pyodide, Скачать файл
# с датасетом, назвать его forest_fires.csv, положить его в View -> File Browser.
# Можно запускать
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Загружаем датасет
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
df = pd.read_csv(url)

# Посмотрим, какие признаки категориальные:
print(df.dtypes)
print("        ")

# Категориальными являются month, day, выкинем их
df_num = df.drop(columns=["month", "day"])

# Применим логарифмическое преобразование как указано в информации о датасете
df_num['area_log'] = np.log1p(df_num['area'])
# Построим линейную регрессию

# Признаки
features = df_num.drop(columns=['area', 'area_log'])
# Цель
target = df_num['area_log']

# Разделим на тестовые данные и данные для обучения
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)

target_test_original = np.expm1(target_test)  # Обратное преобразование для тестовых данных

# Обучаем модель линейной регрессии
model_lr = LinearRegression()
model_lr.fit(features_train, target_train)

# Предсказываем
target_pred_log = model_lr.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)
print ("-----------------------------------------------")
print("Линейная регрессия")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9682.017068063538, RMSE = 98.3972, R² = -0.0219
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.

# Теперь будем строить решение методом бустинга.

# Обучаем модель градиентным бустингом
grb = GradientBoostingRegressor(random_state=42)
grb.fit(features_train, target_train)

# Предсказываем
target_pred_log = grb.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)

print ("-----------------------------------------------")
print ("Градиентный бустинг")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9665.818787831624, RMSE = 98.3149, R² = -0.0202
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.
# Будем подбирать оптимальные параметры

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 2, 3], # Стандартные значения
    'learning_rate': [0.05, 0.1, 0.2] # Стандартные значения
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5, # Делим train на 5 частей, 4 -> обучение, 1 -> проверка
    scoring='neg_mean_squared_error', # Перебираем все комбинации параметров и выбираем оптимальные значения по метрике MSE
    n_jobs=-1 # Обучаемся на всех потоках.
)

# Обучаем модель на оптимальных параметрах
grid_search.fit(features_train, target_train)

# Посмотрим на оптимальные параметры и применим их в нашей моделе
print("Оптимальные параметры:", grid_search.best_params_)
opti_model = grid_search.best_estimator_

# Посчитаем метрики с оптимальными параметрами
target_pred_opti_log = opti_model.predict(features_test)
target_pred_opti = np.expm1(target_pred_opti_log)
mse_opti = mean_squared_error(target_test_original, target_pred_opti)
rmse_opti = np.sqrt(mse_opti)
r2_opti = r2_score(target_test_original, target_pred_opti)

print ("-----------------------------------------------")
print("Градиентный бустинг с оптимальными параметрами")
print (f"Оптимальный MSE: {mse_opti}")
print(f"Оптимальный RMSE = {rmse_opti:.4f}")
print(f"Оптимальный R²   = {r2_opti:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9679.681994809138, RMSE = 98.3854, R² = -0.0217
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.

# Теперь применим one hot encoding для категориальных признаков

df_ohe = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
df_ohe['area_log'] = np.log1p(df_ohe['area'])

# Признаки
features = df_ohe.drop(columns=['area', 'area_log'])
# Цель
target = df_ohe['area_log']

# Разделим на тестовые данные и данные для обучения
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)

target_test_original = np.expm1(target_test)  # Обратное преобразование для тестовых данных

# Обучаем модель линейной регрессии
model_lr.fit(features_train, target_train)

# Предсказываем
target_pred_log = model_lr.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)
print ("-----------------------------------------------")
print("Линейная регрессия с OHE")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9656.18774285676, RMSE = 98.2659, R² = -0.0192
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.

# Теперь будем строить решение методом бустинга.

# Обучаем модель градиентным бустингом
grb.fit(features_train, target_train)

# Предсказываем
target_pred_log = grb.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)

print ("-----------------------------------------------")
print ("Градиентный бустинг c OHE")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9688.284886392059, RMSE = 98.4291, R² = -0.0226
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.
# Будем подбирать оптимальные параметры

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 2, 3], # Стандартные значения
    'learning_rate': [0.05, 0.1, 0.2] # Стандартные значения
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5, # Делим train на 5 частей, 4 -> обучение, 1 -> проверка
    scoring='neg_mean_squared_error', # Перебираем все комбинации параметров и выбираем оптимальные значения по метрике MSE
    n_jobs=-1 # Обучаемся на всех потоках.
)

# Обучаем модель на оптимальных параметрах
grid_search.fit(features_train, target_train)

# Посмотрим на оптимальные параметры и применим их в нашей моделе
print("Оптимальные параметры:", grid_search.best_params_)
opti_model = grid_search.best_estimator_

# Посчитаем метрики с оптимальными параметрами
target_pred_opti_log = opti_model.predict(features_test)
target_pred_opti = np.expm1(target_pred_opti_log)
mse_opti = mean_squared_error(target_test_original, target_pred_opti)
rmse_opti = np.sqrt(mse_opti)
r2_opti = r2_score(target_test_original, target_pred_opti)

print ("-----------------------------------------------")
print("Градиентный бустинг с оптимальными параметрами OHE")
print (f"Оптимальный MSE: {mse_opti}")
print(f"Оптимальный RMSE = {rmse_opti:.4f}")
print(f"Оптимальный R²   = {r2_opti:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9673.816546478298, RMSE = 98.3556, R² = -0.0211
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.
# Теперь сделаем target encoding

df_target = df.copy()
df_target['area_log'] = np.log1p(df['area'])

features = df_target.drop(columns=['area', 'area_log'])
target = df_target['area_log']

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=42
)

global_mean = target_train.mean()

#Для month
month_mean = target_train.groupby(features_train['month']).mean()

features_train['month_te'] = features_train['month'].map(month_mean)
features_test['month_te'] = features_test['month'].map(month_mean)

features_train['month_te'] = features_train['month_te'].fillna(global_mean)
features_test['month_te']  = features_test['month_te'].fillna(global_mean)

# Для day
month_day = target_train.groupby(features_train['day']).mean()

features_train['day_te'] = features_train['day'].map(month_day)
features_test['day_te'] = features_test['day'].map(month_day)

features_train['day_te'] = features_train['day_te'].fillna(global_mean)
features_test['day_te']  = features_test['day_te'].fillna(global_mean)

# Теперь убираем категориальные признаки
features_train = features_train.drop(columns=['month', 'day'])
features_test = features_test.drop(columns=['month', 'day'])

target_test_original = np.expm1(target_test)  # Обратное преобразование для тестовых данных

# Обучаем модель линейной регрессии
model_lr = LinearRegression()
model_lr.fit(features_train, target_train)

# Предсказываем
target_pred_log = model_lr.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)
print ("-----------------------------------------------")
print("Линейная регрессия с TE")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9684.84606443, RMSE = 98.4116, R² = -0.0222
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.

# Теперь будем строить решение методом бустинга.

# Обучаем модель градиентным бустингом
grb = GradientBoostingRegressor(random_state=42)
grb.fit(features_train, target_train)

# Предсказываем
target_pred_log = grb.predict(features_test)

# Обратное преобразование для получения предсказаний в исходной шкале
target_pred = np.expm1(target_pred_log)  # exp(x) - 1

# Посчитаем MSE (средний квадрат ошибки), RMSE (корень из среднего квадрата ошибки), R² (коэффициент детерминации)
mse = mean_squared_error(target_test_original, target_pred)
rmse = np.sqrt(mse)
r2 = r2_score(target_test_original, target_pred)

print ("-----------------------------------------------")
print ("Градиентный бустинг с TE")
print (f"MSE: {mse}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9678.310018331262, RMSE = 98.3784, R² = -0.0215
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.
# Будем подбирать оптимальные параметры

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 2, 3], # Стандартные значения
    'learning_rate': [0.05, 0.1, 0.2] # Стандартные значения
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5, # Делим train на 5 частей, 4 -> обучение, 1 -> проверка
    scoring='neg_mean_squared_error', # Перебираем все комбинации параметров и выбираем оптимальные значения по метрике MSE
    n_jobs=-1 # Обучаемся на всех потоках.
)

# Обучаем модель на оптимальных параметрах
grid_search.fit(features_train, target_train)

# Посмотрим на оптимальные параметры и применим их в нашей моделе
print("Оптимальные параметры:", grid_search.best_params_)
opti_model = grid_search.best_estimator_

# Посчитаем метрики с оптимальными параметрами
target_pred_opti_log = opti_model.predict(features_test)
target_pred_opti = np.expm1(target_pred_opti_log)
mse_opti = mean_squared_error(target_test_original, target_pred_opti)
rmse_opti = np.sqrt(mse_opti)
r2_opti = r2_score(target_test_original, target_pred_opti)

print ("-----------------------------------------------")
print("Градиентный бустинг с оптимальными параметрами TE")
print (f"Оптимальный MSE: {mse_opti}")
print(f"Оптимальный RMSE = {rmse_opti:.4f}")
print(f"Оптимальный R²   = {r2_opti:.4f}")
print ("-----------------------------------------------")
print("        ")
# Получили значения MSE: = 9668.8464409964, RMSE = 98.3303, R² = -0.0205
# RMSE говорит нам о том, что в среднем модель ошибается на 98 гектаров при предсказании площади выгоревшего леса.
# Коэффициент детерминации < 0, ошибка модели больше дисперсии.
# Точность OHE и TE на этом датасете практически одинакова

# Визуализируем

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
sc = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=target_train,
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(sc, label='area_log')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Объекты в пространстве первых двух главных компонент')
plt.grid(True)
plt.show()

lr_pca = LinearRegression()
lr_pca.fit(X_pca, target_train)

gb_pca = GradientBoostingRegressor(random_state=42)
gb_pca.fit(X_pca, target_train)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z_lr = lr_pca.predict(grid).reshape(xx.shape)
Z_gb = gb_pca.predict(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z_lr, cmap='viridis', alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_train, cmap='viridis', edgecolor='k', s=20)
plt.colorbar(label='area_log')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Линейная регрессия в PCA-пространстве')
plt.show()

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z_gb, cmap='viridis', alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_train, cmap='viridis', edgecolor='k', s=20)
plt.colorbar(label='area_log')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Градиентный бустинг в PCA-пространстве')
plt.show()
