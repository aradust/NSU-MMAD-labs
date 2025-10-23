from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from config import RANDOM_STATE


def get_models():
    """
    Возвращает словарь с моделями для классификации.

    Returns:
        dict: словарь с моделями {имя: модель}
    """
    models = {
        "LDA (линейный)": LinearDiscriminantAnalysis(),
        "QDA (квадратичный)": QuadraticDiscriminantAnalysis(),
        "Логистическая регрессия": LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
        "SVM (линейное ядро)": SVC(kernel='linear', random_state=RANDOM_STATE),
        "SVM (квадратичное ядро)": SVC(kernel='poly', degree = 2, random_state=RANDOM_STATE)
    }
    return models


def train_lda(X, y):
    """
    Обучает модель LDA на всех признаках.

    Args:
        X (DataFrame): признаки
        y (Series): целевая переменная

    Returns:
        tuple: (обученная модель, предсказания)
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)
    return lda, y_pred