# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sns.set()
df = pd.read_csv('multiTimeline.csv', skiprows=1)
# шапка csv
print(df.head())
print("\n<---------------------------->\n")


df.columns = ['month', 'diet', 'gym', 'finance']
# шапка csv с выбранными колонками
print(df.head())
print("\n<---------------------------->\n")

df.month = pd.to_datetime(df.month)
# шапка csv с выбранными колонками и месяцами по индексу
df.set_index('month', inplace=True)
print(df.head())
print("\n<---------------------------->\n")

# визуализация данных в виде 3-х линейных графиков «диета», «спортзал» и «финансы»
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.show()

# передискретизация, сглаживание, управление окнами, скользящее среднее
diet = df['diet']
diet_resample = diet.resample('A').mean()
diet_rolling = diet.rolling(12).mean()

ax = diet.plot(figsize=(20,10), linewidth=5, fontsize=20, alpha=0.5, style='-')
diet_resample.plot(style=':', linewidth=5, label='Resample at year frequency', ax=ax)
diet_rolling.plot(style='--', linewidth=5, label='Rolling average', ax=ax)
ax.legend(fontsize=20)
# diet.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
# diet.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);
plt.show()

#Сглаживание с помощью библиотеки NumPy
x = np.asarray(df[['diet']])
win_size = 12
win_half = int(win_size / 2)
diet_smooth = np.array([x[(idx-win_half):(idx+win_half)].mean() for idx in np.arange(win_half, len(x))])
plt.plot(diet_smooth)


# Созадём новый датафрейм с данными о диете и спортзале
gym = df['gym']
df_average = pd.concat([diet.rolling(12).mean(), gym.rolling(12).mean()], axis=1)
df_average.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
plt.show()
df_diff_trend = df[['diet', 'gym']] - df_average
df_diff_trend.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
plt.show()

# Разница первого порядка: сезонные модели
assert np.all((diet.diff() == diet - diet.shift())[1:])
df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
plt.show()

# Периодичность и корреляция
df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
print(df.corr())
plt.show()
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

# корреляция
df.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year')
print(df.diff().corr())
plt.show()

# матрица корреляции
sns.heatmap(df.diff().corr(), cmap='coolwarm')
plt.show()

# Разложение временных рядов по тренду, сезонности и остаткам
from statsmodels.tsa.seasonal import seasonal_decompose
x = gym
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

# Автокорреляция
from pandas.plotting import autocorrelation_plot
x = df["diet"].astype(float)
autocorrelation_plot(x)
plt.show()

from statsmodels.tsa.stattools import acf
x_diff = x.diff().dropna() # first item is NA
lag_acf = acf(x_diff, nlags=36)
plt.plot(lag_acf)
plt.title('Autocorrelation Function')
plt.show()

from statsmodels.tsa.stattools import acf, pacf
x = df["gym"].astype(float)
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
plt.title('Autocorrelation Function (q=1)')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(x_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function (p=1)')
plt.tight_layout()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(x, order=(1, 1, 1)).fit() # fit model
print(model.summary())
plt.figure(figsize=(20,10))
plt.plot(x)
plt.plot(model.predict(), color='red')
plt.title('RSS: %.4f'% sum((model.fittedvalues-x)**2))
plt.show()
