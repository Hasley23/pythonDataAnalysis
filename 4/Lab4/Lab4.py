# библиотеки
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# функция создания описательной статистики
def set_up_statistic(marks, ax):
    # Среднее арифметическое
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.90,
            'Среднее арифметическое: {0}'.format(np.mean(marks)), fontsize=9)
    # Медиана
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.85,
            'Медиана: {0}'.format(np.median(marks)), fontsize=9)
    # Мода
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.80,
            'Мода: {0}'.format(stats.mode(marks)[0]), fontsize=9)
    # Среднее геометрическое
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.75,
            'Среднее геометрическое: {0}'.format(stats.hmean(marks)), fontsize=9)
    # Размах
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.70,
            'Размах: {0}'.format(np.ptp(marks)), fontsize=9)
    # Межквартальный размах
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.65,
            'Межквартальный размах: {0}'.format(stats.iqr(marks)), fontsize=9)
    # Межквантильный диапазон (Интердециальный размах)
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.60,
            'Интердециальный размах: {0}'.format(stats.iqr(marks, rng=(10, 90))), fontsize=9)
    # Дисперсия
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.55,
            'Дисперсия: {0}'.format(stats.variation(marks)), fontsize=9)
    # Среднеквадратичное отклонение
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.50,
            'Среднеквадратичное отклонение: {0}'.format(np.std(marks)), fontsize=9)
    # Коэффициент ассиметрии
    ax.text(ax.viewLim.intervalx.max(), ax.viewLim.intervaly.max() * 0.45,
            'Коэффициент ассиметрии : {0}'.format(stats.skew(marks)), fontsize=9)

# Файл csv
data = pd.read_csv('marks_groups.csv')
# Ненужный столбец
data = data.drop('Группа', axis=1)
# Колонки
subjects = [column for column in data.columns][:]

# Создвние диаграмм для всех дисциплин
for subject in subjects:
    subject_data = data[[subject]]

    # Построение распределения    
    g = sns.displot(subject_data, x=subject, binwidth=4, height=4,
                        facet_kws=dict(margin_titles=True), kde=True, color = 'green')
    subject_marks = np.array([mark for mark in subject_data[subject]])
    # Настройка удобного отображения
    set_up_statistic(subject_marks, g.ax)
    # Вывод
    g.savefig('./diagrams/{0}.svg'.format(subject))
       

