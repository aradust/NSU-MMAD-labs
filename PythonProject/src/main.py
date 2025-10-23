
from config import setup_plotting_style, SELECTED_FEATURES
from data_loader import load_wine_data
from data_printer import print_data_info
from visualization import plot_pairwise_distributions, plot_decision_boundaries
from models import get_models
from evaluation import evaluate_models
from lda_analysis import perform_lda_analysis

def main():
    # Настройка стиля для графиков
    setup_plotting_style()

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

    # Выбор признаков
    feature1, feature2 = SELECTED_FEATURES
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

if __name__ == "__main__":
    main()

