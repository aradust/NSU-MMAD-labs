import pandas as pd
from sklearn.datasets import load_wine

def load_wine_data():
    """
    Загружает набор данных о винах и возвращает DataFrame.

    Returns:
        tuple: (DataFrame с данными, объект с метаданными)
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target

    return df, wine

