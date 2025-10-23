from models import train_lda
from visualization import plot_lda_results
from evaluation import evaluate_lda
from config import FIGSIZE_HUGE, DPI


def perform_lda_analysis(df, wine):
    """
    Проводит анализ с использованием LDA.

    Args:
        df (DataFrame): DataFrame с данными
        wine: объект с метаданными набора данных

    Returns:
        tuple: (DataFrame с предсказаниями, точность, матрица ошибок)
    """

    # Подготовка данных
    X_full = df.drop('target', axis=1)
    y = df['target']

    # Обучение LDA на всех признаках
    lda, y_pred = train_lda(X_full, y)

    # Добавление предсказаний в DataFrame
    df['lda_pred'] = y_pred

    # Визуализация результатов
    plot_lda_results(df, wine, figsize=FIGSIZE_HUGE, dpi=DPI)

    # Анализ ошибок LDA
    lda_accuracy, cm = evaluate_lda(y, y_pred, wine)

    return df, lda_accuracy, cm