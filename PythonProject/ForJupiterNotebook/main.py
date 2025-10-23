# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score

# Константы
RANDOM_STATE = 42
CV_FOLDS = 5


# Функция загрузки данных
def load_wine_data():
    """Загружает набор данных о винах и возвращает DataFrame."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine


# Функция вывода информации о данных
def print_data_info(df, wine):
    """Выводит основную информацию о наборе данных."""
    print(f"Размер данных: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nКлассы вин:", wine.target_names)


# Функция визуализации попарных распределений
def plot_pairwise_distributions(df, wine):
    """Строит попарные распределения признаков по классам вин."""
    plt.figure(figsize=(20, 20))
    sns.pairplot(df, hue='target', vars=wine.feature_names,
                 markers=['o', 's', 'D'], diag_kind='kde')
    plt.suptitle('Попарные распределения признаков по классам вин', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=300)
    plt.show()


# Функция визуализации разделяющих границ
def plot_decision_boundary(model, X, y, feature1, feature2, title):
    """Визуализирует разделяющие границы для модели классификации."""
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Создаем DataFrame с именами столбцов для предсказания
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=X.columns)

    Z = model.predict(grid_df)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold,
                          edgecolor='k', s=40)

    plt.legend(handles=scatter.legend_elements()[0],
               labels=['Class 0', 'Class 1', 'Class 2'],
               title="Истинные классы")
    plt.title(title)
    plt.xlabel(feature1)
    plt.ylabel(feature2)


# Функция визуализации разделяющих границ для всех моделей
def plot_decision_boundaries(models, X, y, feature1, feature2):
    """Строит разделяющие границы для нескольких моделей."""
    plt.figure(figsize=(15, 12))
    plt.suptitle('Разделяющие границы для разных методов', y=1.02, fontsize=16)

    for i, (name, model) in enumerate(models.items(), 1):
        model.fit(X, y)
        plt.subplot(2, 3, i)
        plot_decision_boundary(model, X, y, feature1, feature2, name)

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=300)
    plt.show()


# Функция получения моделей
def get_models():
    """Возвращает словарь с моделями для классификации."""
    return {
        "LDA (линейный)": LinearDiscriminantAnalysis(),
        "QDA (квадратичный)": QuadraticDiscriminantAnalysis(),
        "Логистическая регрессия": LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
        "SVM (линейное ядро)": SVC(kernel='linear', random_state=RANDOM_STATE),
        "SVM (квадратичное ядро)": SVC(kernel='poly', degree=2, random_state=RANDOM_STATE)
    }


# Функция обучения LDA
def train_lda(X, y):
    """Обучает модель LDA на всех признаках."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)
    return lda, y_pred


# Функция оценки моделей
def evaluate_models(models, X_two, X_full, y):
    """Оценивает качество моделей с помощью кросс-валидации."""
    print("\nОценка качества классификации (accuracy):")
    print("-" * 60)

    results = []
    for name, model in models.items():
        # Оценка на двух признаках
        scores_two = cross_val_score(model, X_two, y, cv=CV_FOLDS)
        # Оценка на всех признаках
        scores_full = cross_val_score(model, X_full, y, cv=CV_FOLDS)

        results.append({
            'Метод': name,
            'Два признака': f"{scores_two.mean():.3f} ± {scores_two.std():.3f}",
            'Все признаки': f"{scores_full.mean():.3f} ± {scores_full.std():.3f}"
        })

        print(f"{name}:")
        print(f"  На двух признаках: {scores_two.mean():.3f} ± {scores_two.std():.3f}")
        print(f"  На всех признаках: {scores_full.mean():.3f} ± {scores_full.std():.3f}\n")

    # Сохранение результатов в таблицу
    results_df = pd.DataFrame(results)
    print("\nСводная таблица результатов:")
    print(results_df)
    results_df.to_csv('classification_results.csv', index=False)

    return results_df


# Функция оценки LDA
def evaluate_lda(y_true, y_pred, wine):
    """Оценивает качество работы LDA."""
    cm = confusion_matrix(y_true, y_pred)
    lda_accuracy = accuracy_score(y_true, y_pred)

    print("\nАнализ работы LDA:")
    print(f"Точность LDA на обучающей выборке: {lda_accuracy:.3f}")
    print("\nМатрица ошибок:")
    print(cm)

    # Сохранение матрицы ошибок
    cm_df = pd.DataFrame(cm,
                         index=wine.target_names,
                         columns=wine.target_names)
    cm_df.to_csv('lda_confusion_matrix.csv')

    return lda_accuracy, cm


# Функция визуализации результатов LDA
def plot_lda_results(df, wine):
    """Визуализирует результаты работы LDA."""
    pairplot = sns.pairplot(
        df,
        vars=wine.feature_names,
        hue='lda_pred',
        markers=['o', 's', 'D'],
        diag_kind='kde'
    )

    # Настройка внешнего вида
    pairplot.fig.suptitle(
        'Истинные классы (форма маркера) vs Предсказания LDA (цвет)\n' +
        'Во всех двумерных проекциях',
        y=1.02,
        fontsize=18
    )

    for ax in pairplot.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right')
        ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig('lda_visualization.png', dpi=300)
    plt.show()


# Функция анализа LDA
def perform_lda_analysis(df, wine):
    """Проводит анализ с использованием LDA."""
    print("\nЭтап 3: LDA на всех признаках и визуализация")

    # Подготовка данных
    X_full = df.drop('target', axis=1)
    y = df['target']

    # Обучение LDA на всех признаках
    lda, y_pred = train_lda(X_full, y)

    # Добавление предсказаний в DataFrame
    df['lda_pred'] = y_pred

    # Визуализация результатов
    plot_lda_results(df, wine)

    # Анализ ошибок LDA
    lda_accuracy, cm = evaluate_lda(y, y_pred, wine)

    return df, lda_accuracy, cm


# Основная функция
def main():
    # Установка стиля для графиков
    sns.set(style="whitegrid", palette="viridis")

    # =============================================
    # ЭТАП 1: ЗАГРУЗКА ДАННЫХ И ВИЗУАЛИЗАЦИЯ
    # =============================================
    print("Этап 1: Загрузка данных и визуализация")

    # Загрузка данных
    df, wine = load_wine_data()

    # Вывод информации о данных
    print_data_info(df, wine)

    # Визуализация попарных распределений
    plot_pairwise_distributions(df, wine)

    # =============================================
    # ЭТАП 2: ВЫБОР ПРИЗНАКОВ И КЛАССИФИКАЦИЯ
    # =============================================
    print("\nЭтап 2: Классификация и оценка качества")

    # Выбор двух признаков
    feature1 = 'alcohol'
    feature2 = 'color_intensity'
    X_two = df[[feature1, feature2]]
    y = df['target']
    X_full = df.drop('target', axis=1)

    # Инициализация моделей
    models = get_models()

    # Визуализация разделяющих границ
    plot_decision_boundaries(models, X_two, y, feature1, feature2)

    # Оценка качества кросс-валидацией
    results_df = evaluate_models(models, X_two, X_full, y)

    # =============================================
    # ЭТАП 3: LDA НА ВСЕХ ПРИЗНАКАХ И ВИЗУАЛИЗАЦИЯ
    # =============================================
    df, lda_accuracy, cm = perform_lda_analysis(df, wine)

    print("\nАнализ завершен! Результаты сохранены в файлах:")
    print("- pairplot.png: визуализация попарных распределений")
    print("- decision_boundaries.png: разделяющие границы классификаторов")
    print("- lda_visualization.png: визуализация работы LDA")
    print("- classification_results.csv: таблица с результатами классификации")
    print("- lda_confusion_matrix.csv: матрица ошибок LDA")


# Запуск основной функции
if __name__ == "__main__":
    main()