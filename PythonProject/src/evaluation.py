import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from config import CV_FOLDS


def evaluate_models(models, X_two, X_full, y):
    """
    Оценивает качество моделей с помощью кросс-валидации.

    Args:
        models (dict): словарь с моделями {имя: модель}
        X_two (DataFrame): два выбранных признака
        X_full (DataFrame): все признаки
        y (Series): целевая переменная

    Returns:
        DataFrame: таблица с результатами оценки
    """

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


    # Сохранение результатов в таблицу
    results_df = pd.DataFrame(results)
    results_df.to_csv('../TablesAndPictures/classification_results.csv', index=False)

    return results_df


def evaluate_lda(y_true, y_pred, wine):
    """
    Оценивает качество работы LDA.

    Args:
        y_true (Series): истинные значения
        y_pred (array): предсказанные значения
        wine: объект с метаданными набора данных

    Returns:
        tuple: (точность, матрица ошибок)
    """
    cm = confusion_matrix(y_true, y_pred)
    lda_accuracy = accuracy_score(y_true, y_pred)

    # Сохранение матрицы ошибок
    cm_df = pd.DataFrame(cm,
                         index=wine.target_names,
                         columns=wine.target_names)
    cm_df.to_csv('../TablesAndPictures/lda_confusion_matrix.csv')

    return lda_accuracy, cm