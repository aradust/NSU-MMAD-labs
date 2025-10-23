Инструкция по запуску кода через весию Jupiter Notebook в браузере.

1) Открыть браузерную версию Jupiter notebook
![img.png](img.png)
![img_1.png](img_1.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_5.png](img_5.png)
2) Выбрать ядро Python (Pyodide)
![img_6.png](img_6.png)
3) В ячейку прописать следующую команду и запустить:
%pip install seaborn pandas numpy matplotlib scikit-learn
![img_7.png](img_7.png)
4) Создать новую ячейку
![img_8.png](img_8.png)
5) Скопировать код из файла PythonProject/ForJupiterNotebook/main.py
6) Вставить его в ячейку и запустить.
![img_9.png](img_9.png)
7) Дождаться окончания работы программы, после чего открыть View -> File Browser.
![img_10.png](img_10.png)
8) В открывшемся окне найти следующие файлы:
*  pairplot.png - визуализация распределения классов на всех парах переменных .
* decision_boundaries.png - разделяющая кривая решения методами линейный и квадратичный дискриминант, логистическая регрессия, SVM (линейное и квадратичное ядро) (две переменные).
* classification_results.csv - Таблица с оценками качества классификации этими методами при помощи кросс-валидации на выбранных признаках и во всем признаковом пространстве.
* lda_visualization.png - визуализация ответов алгоритма линейного дискриминанта на всех переменных.
* lda_confusion_matrix.csv - матрица ошибок модели линейного дискриминанта.

Инструкция по запуску кода через IDE Pycharm Community

1) Склонировать репозиторий с гитхаба командой git clone https://github.com/aradust/NSU-MMAD-labs.git
2) Открыть проект в IDE Pycharm Community.
3) Запустить файл PythonProject/src/main.py
4) Аналогичые результаты можно найти в директории PythonProject/TablesAndPictures

Все файлы с исходным кодом находятся в директории PythonProject/src
В директории PythonProject/examples находятся примеры написания кода и выполнения аналогичных заданий.
В директории PythonProject/Litherature собраны некоторые теоретические материалы.
В директории PythonProject/TZ располагается описание данного задания.