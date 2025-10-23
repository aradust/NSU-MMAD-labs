import pandas as pd

def print_data_info(df, wine):
    """
    Выводит основную информацию о наборе данных.

    Args:
        df (DataFrame): DataFrame с данными
        wine: объект с метаданными набора данных
    """
    pd.set_option('display.max_columns', None)
    print("\nТаблица:")
    print(df)