import csv
import random
from copy import deepcopy
from math import sqrt
from itertools import groupby
import matplotlib.pyplot as plt
from matplotlib import rcParams as config

config['figure.figsize'] = 12, 20
config['axes.labelsize'] = 9
config.update({'figure.autolayout': True})
config['legend.loc'] = 'upper right'


class Vine:
    # Веса параметров для получения лучшей точности определения
    weight = {
        'fixed_acidity': 1,
        'volatile_acidity': 1,
        'citric_acid': 1,
        'residual_sugar': 1,
        'chlorides': 1,
        'free_sulfur_dioxide': 1,
        'total_sulfur_dioxide': 1,
        'density': 1,
        'pH': 1,
        'sulphates': 1,
        'alcohol': 1,
    }

    # Цвет для кадого класса (для графиков)
    palette = {
        3: '#ff0000',
        4: '#ff6600',
        5: '#ffc800',
        6: '#5eff00',
        7: '#00c8ff',
        8: '#001eff',
        9: '#560094',
    }

    def __init__(self, fixed_acidity=0, volatile_acidity=0, citric_acid=0,
                    residual_sugar=0, chlorides=0, free_sulfur_dioxide=0,
                    total_sulfur_dioxide=0, density=0, pH=0, sulphates=0,
                    alcohol=0, quality=0):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.quality = int(quality)

    @staticmethod
    def from_line(record):
        '''Заполнение объекта вина из строки csv'''
        return Vine(*[float(num) for num in list(record)])

    @staticmethod
    def fields():
        '''Возвращает список имён всех полей класса'''
        fields = list(vars(Vine()))
        fields.remove('quality')
        return fields

    def attr(self, field):
        '''Возвращает (ссылку на) поле класса по имени'''
        return getattr(self, field)

    @staticmethod
    def set_weights(values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
        '''Заполнение массива весов:'''
        Vine.weight = dict(zip(Vine.fields(), values))

    @staticmethod
    def set_weight_only(field):
        '''Для всех параметров, кроме одного, устанавливается вес = 0, поэтому они не учитываются при расчёте'''
        Vine.set_weights()
        Vine.weight[field] = 1

    def distance(self, other):
        '''Расчитываем расстояние (по параметрам) между двумя винами'''
        def d(field): return self.weight[field] * \
                (self.attr(field) - other.attr(field))**2
        return sqrt(sum(d(field) for field in Vine.fields()))

    def change_quality(self, quality=0):
        '''Возвращяет копию объекта с изменённым качеством'''
        copy_vine = deepcopy(self)
        copy_vine.quality = quality
        return copy_vine

    def __str__(self):
        return str( [ self.attr(f) for f in Vine.fields() ] + [ self.quality ])


# Группировка
def group(array):
    '''Разделение массива объектов на группы по качеству'''
    def keyfunc(v): return (v if isinstance(v, Vine) else v[0]).quality
    return groupby(sorted(array, key=keyfunc), key=keyfunc)


# Загрузка csv
def load_dataset(filename):
    '''Загрузка данных из файла'''
    with open(filename, 'r') as csvfile:
        record_list = list(csv.reader(csvfile, delimiter=';'))
        del record_list[0]
    return [Vine.from_line(record) for record in record_list]

# Функция нормализации
def normalize(data):
    ''''''
    def calc_stats(col):
        m = sum(col) / len(col)
        D = sum( (x - m) ** 2 for x in col ) / (len(col) - 1)
        return m, D ** (1/2)

    def norm(vine, field):
        m, d = stats[field]
        return (vine.attr(field) - m) / d

    fields = Vine.fields()
    # Вычисление мат. ожидания и среднеквадратичного отклонения для каждого параметра вина
    stats = { f: calc_stats([v.attr(f) for v in data]) for f in fields }

    return [ Vine( *[ norm(v, f) for f in fields ], v.quality) for v in data]

def train_test_split(all, *, onTraining=0.95, ofSet=1):
    '''Разделение данных на тренировочные и тестовые'''
    dataset = all
    if ofSet < 1:
        dataset, _ = train_test_split(dataset, onTraining=ofSet)
    trainCount = int(len(dataset) * onTraining)
    train = random.sample(dataset, trainCount)
    test = [vine for vine in dataset if not vine in train]
    return train, test

# Метод k-ближайшего соседа
def knn(vine, vines, k=3):
    '''Метод k-ближайшего соседа'''

    def result_vote(nearest):
        '''Определённое голосованием качество нового вина'''
        def weight(distance): return 1 / (distance ** 2) if distance != 0 else 200
        def sum_weight(array): return sum(weight(d) for _, d in list(array))

        weihts = [(k, sum_weight(g)) for k, g in group(nearest)]
        votes = sorted(weihts, key=lambda item: item[1], reverse=True)
        return votes[0][0]

    # Массив расстояний от тестового вина до известных вин
    distances = [vine.distance(other_vine) for other_vine in vines]
    # Сортировка вин по близости к тестовому
    neighbours = sorted(zip(vines, distances), key=lambda item: item[1])
    # k ближайших вин
    nearest = neighbours[0:k]

    return vine.change_quality(result_vote(nearest))

def get_accuracy(defined, original):
    '''Точность определения качества вина'''
    matched = [int(d.quality == o.quality) for d, o in zip(defined, original)]
    return (sum(matched) / len(matched)) * 100


def calculate_weight(train, test, k):
    '''Расчёт оптимальных весов'''
    def coeff(acc):
        return  0.01 if acc < 30 else \
                0.35 if acc < 34 else \
                0.75 if acc < 60 else \
                1

    fields = Vine.fields()
    countFields = len(fields)

    fig = plt.figure()
    graph = [fig.add_subplot(countFields, 1, i+1) for i in range(countFields)]

    calculateWeight = []

    for g, field in zip(graph, fields):
        Vine.set_weight_only(field)
        defined = [knn(vine, train, k) for vine in test]

        accuracy = get_accuracy(defined, test)
        print(f'Accuracy by only {field}: {accuracy:.2f}%')
        calculateWeight.append(accuracy * coeff(accuracy))

        f = [field, 'quality']
        Plot.set_description(g, f'Accuracy: {accuracy:.2f} %', 'quality', field)
        markers = Plot.correct(g, test, defined, f)

    fig.legend(markers, ['Correct', 'Uncorrect'])

    fig.savefig('Vine - calculateWeight.svg')

    plt.close()

    return calculateWeight


# Графики
class Plot:
    params = {
        'point': {
            'marker': 'o',
        },
        'train_3d': {
            'alpha': 0.3,
        },
        'define_3d': {
            'marker': '*',
            'alpha': 0.9,
            's': 160,
        },
        'correct':{
            'label': 'Correct',
            'color': '#1B5E20',
        },
        'uncorrect': {
            'label': 'Uncorrect',
            'color': '#D50000',
        },
    }

    @staticmethod
    def set_description(plot, title, x, y, z=''):
        plot.set_title(title, fontsize=9)
        plot.set_xlabel(x)
        plot.set_ylabel(y)
        if z: plot.set_zlabel(z)

    @staticmethod
    def coord(vines, fields):
        '''Возвращает массивы координат'''
        return zip( * [ [v.attr(field) for field in fields] for v in vines ])

    @staticmethod
    def points(ax, vines, fields, options={}):
        '''Вывод точек вин (вина одного качества - одним цветом)'''
        total_options = { **Plot.params['point'], **options }

        return [ ax.scatter(*Plot.coord(list(g), fields), **total_options, c=Vine.palette[k])
                        for k, g in group(vines)]

    @staticmethod
    def training(train):
        '''Вывод вин по каждому из параметров, чтобы посмотреть, насколько хорошо делятся вина по качеству'''
        fields = Vine.fields()
        countFields = len(fields)

        fig = plt.figure()
        graph = [fig.add_subplot(countFields, 1, i+1) for i in range(countFields)]

        for g, field in zip(graph, fields):
            Plot.set_description(g, '', field, 'quality')
            f = [field, 'quality']
            markers = Plot.points(g, train, f)

        fig.legend(markers, Vine.palette.keys(), title='Качество')
        fig.savefig('Vine - train.svg')
        plt.close()

    @staticmethod
    def _3d(train, test, define):
        '''Построение трёхмерного графика вин'''

        def add_legend(markers, labels, title, loc):
            plt.gca().add_artist(plt.legend(markers,
                                    labels, title=title, loc=loc, fontsize=12))

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        max_accuracy_fields = list(dict(sorted(Vine.weight.items(), key=lambda item: item[1], reverse=True)).keys())[:2]
        f = ['quality', *max_accuracy_fields]
        Plot.set_description(ax, '', *f)

        markersTrain  = Plot.points(ax, train,  f, Plot.params['train_3d'])
        add_legend(markersTrain, Vine.palette.keys(), 'Качество', 'upper left')

        # markersDefine = Plot.points(ax, define, f, Plot.params['define_3d'])
        # add_legend(markersDefine, Vine.palette.keys(), 'Опр. качество', 'upper right')

        markersDefine = Plot.correct(ax, test, define, f, Plot.params['define_3d'])
        add_legend(markersDefine, ['Correct', 'Uncorrect'], 'Опр. качество', 'upper right')

        plt.savefig('Vine - 3d.svg')
        plt.show()

    @staticmethod
    def correct(graph, test, defined, fields, options={}):
        '''Правильно и неправильно определённые вина'''
        correct = [ [], [] ]
        for d, o in zip(defined, test):
            correct[d.quality == o.quality].append(d)
        markers = [
            graph.scatter(*Plot.coord(correct[True],  fields), **{ **options, **Plot.params['correct']}),
            graph.scatter(*Plot.coord(correct[False], fields), **{ **options, **Plot.params['uncorrect']}),
        ]
        return markers


# Тояка входа
def main():
    # Загрузка данных и деление их на тренировочные и тестовые
    vines = load_dataset('./winequality-unique.csv')
    train, test = train_test_split(vines, onTraining=0.90, ofSet=0.20)

    Plot.training(train)

    # Определение качества вин из тестового множества
    k = 3

    Vine.set_weights(calculate_weight(train, test, k))

    defined = [knn(vine, train, k) for vine in test]
    print(f'Total accuracy: {get_accuracy(defined, test):.2f} %')

    Plot._3d(train, test, defined)



if __name__ == '__main__':
    main()