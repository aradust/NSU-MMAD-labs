import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap


def plot_pairwise_distributions(df, wine, figsize=(20, 20), dpi=300, save_path='../TablesAndPictures/pairplot.png'):
    """
    Строит попарные распределения признаков по классам вин.

    Args:
        df (DataFrame): DataFrame с данными
        wine: объект с метаданными набора данных
        figsize (tuple): размер фигуры
        dpi (int): разрешение изображения
        save_path (str): путь для сохранения изображения
    """
    plt.figure(figsize=figsize)
    sns.pairplot(df, hue='target', vars=wine.feature_names,
                 markers=['o', 's', 'D'], diag_kind='kde')
    plt.suptitle('Попарные распределения признаков по классам вин', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)


def plot_decision_boundary(model, X, y, feature1, feature2, title):
    """
    Визуализирует разделяющие границы для модели классификации.

    Args:
        model: обученная модель классификации
        X (DataFrame): признаки
        y (Series): целевая переменная
        feature1 (str): название первого признака
        feature2 (str): название второго признака
        title (str): заголовок графика
    """
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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


def plot_decision_boundaries(models, X, y, feature1, feature2, figsize=(15, 12),
                             dpi=300, save_path='../TablesAndPictures/decision_boundaries.png'):
    """
    Строит разделяющие границы для нескольких моделей.

    Args:
        models (dict): словарь с моделями {имя: модель}
        X (DataFrame): признаки
        y (Series): целевая переменная
        feature1 (str): название первого признака
        feature2 (str): название второго признака
        figsize (tuple): размер фигуры
        dpi (int): разрешение изображения
        save_path (str): путь для сохранения изображения
    """
    plt.figure(figsize=figsize)
    plt.suptitle('Разделяющие границы для разных методов', y=1.02, fontsize=16)

    for i, (name, model) in enumerate(models.items(), 1):
        model.fit(X, y)
        plt.subplot(2, 3, i)
        plot_decision_boundary(model, X, y, feature1, feature2, name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)


def plot_lda_results(df, wine, figsize=(25, 25), dpi=300, save_path='../TablesAndPictures/lda_visualization.png'):
    """
    Визуализирует результаты работы LDA.

    Args:
        df (DataFrame): DataFrame с данными и предсказаниями
        wine: объект с метаданными набора данных
        figsize (tuple): размер фигуры
        dpi (int): разрешение изображения
        save_path (str): путь для сохранения изображения
    """
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
    plt.savefig(save_path, dpi=dpi)