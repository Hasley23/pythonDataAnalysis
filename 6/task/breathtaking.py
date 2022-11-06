# Подключение библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# инициализация
sns.set()
pd.set_option("display.max_colwidth",15)
# динамика популярности: веб-поиск
df = pd.read_csv('multiTimeline.csv', skiprows=1)
print(df.head())
print("\n<----------------------------------------->\n");

df.columns = ['week', 'Уилл Смит', 'Киану Ривз', 'Джонни Депп']
print(df.head())
print("\n<----------------------------------------------------->\n");

df.week = pd.to_datetime(df.week)
# inplace = true
df.set_index('week', inplace=True)
print(df.head())
print("\n<----------------------------------------------------->\n");

# визуализация
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.show()

# рассчитываем скользящее среднее и передискретизацию
Will = df['Уилл Смит']
Will_resample = Will.resample('A').mean() # передискретизацию
Will_rolling = Will.rolling(12).mean() # скользящее среднее

ax = Will.plot(figsize=(20,10), linewidth=5, fontsize=20, alpha=0.5, style='-')
Will_resample.plot(style=':', linewidth=5, label='Resample at year frequency', ax=ax, color='red')
Will_rolling.plot(style='--', linewidth=5, label='Rolling average', ax=ax, color='blue')
ax.legend(fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.show()

#Сглаживание с помощью библиотеки NumPy
x = np.asarray(df[['Уилл Смит']])
win_size = 12
win_half = int(win_size / 2)
Will_smooth = np.array([x[(idx-win_half):(idx+win_half)].mean() for idx in np.arange(win_half, len(x))])
plt.figure(figsize=(20,10))
plt.plot(Will_smooth, linewidth=5)


#Созадём новый датафрейм с данными о Киану Ривзе и Джонни Деппе
Keanu = df['Киану Ривз']
df_average = pd.concat([Will.rolling(12).mean(), Keanu.rolling(12).mean()], axis=1)
df_average.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')

# Находим вычитание тренда
df_diff_trend = df[['Уилл Смит', 'Киану Ривз']] - df_average
df_diff_trend.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
plt.show()

# Находим разницу первого порядка для сезонной модели
assert np.all((Will.diff() == Will - Will.shift())[1:])
df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')

# Находим переодичность и составляем корреляционную модель
# Отрицательные значения означают, что трендовые компоненты имеют отрицательную корреляцию
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
print(df.corr())

# Корреляционная модель
sns.heatmap(df.corr(), cmap='coolwarm')

print("\n<----------------------------------------------------->\n");

# Корреляция разности первого порядка
df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
print(df.diff().corr())

# Корреляционная модель (первого порялка)
sns.heatmap(df.diff().corr(), cmap='coolwarm')

# проводим разложения на временной ряд по тренду, сезонности и остатку
from statsmodels.tsa.seasonal import seasonal_decompose

x = Keanu
x = x.astype(float) # force float
decomposition = seasonal_decompose(x)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(x, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# пример автокорреляции
from pandas.plotting import autocorrelation_plot
x = df["Уилл Смит"].astype(float)
autocorrelation_plot(x)

# Функция автокорреляции
from statsmodels.tsa.stattools import acf

x_diff = x.diff().dropna() # first item is NA
lag_acf = acf(x_diff, nlags=36)
plt.plot(lag_acf)
plt.title('Autocorrelation Function')


# ACF/PACF
from statsmodels.tsa.stattools import acf, pacf

x = df["Уилл Смит"].astype(float)

x_diff = x.diff().dropna() # first item is NA

# ACF and PACF plots:
lag_acf = acf(x_diff, nlags=20)
lag_pacf = pacf(x_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function  (q=1)')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function (p=1)')
plt.tight_layout()

# создание модели функции ARIMA с параметрами p и q
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(x, order=(1, 1, 1)).fit() # fit model
print(model.summary())
plt.figure(figsize=(20,10))
plt.plot(x, linewidth=3)
plt.plot(model.predict(), color='red', linewidth=5)
plt.title('RSS: %.4f'% sum((model.fittedvalues-x)**2))
plt.show()