import seaborn as sns

# Настройка стиля для графиков
def setup_plotting_style():
    sns.set(style="whitegrid", palette="viridis")

# Константы
RANDOM_STATE = 42
CV_FOLDS = 5
FIGSIZE_LARGE = (20, 20)
FIGSIZE_MEDIUM = (15, 12)
FIGSIZE_HUGE = (25, 25)
DPI = 300

# Выбранные признаки для визуализации
SELECTED_FEATURES = ['alcohol', 'color_intensity']