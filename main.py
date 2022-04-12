#Подлкючаем пакет необходимых для работы библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#Считываем исходные данные
dataset = pd.read_excel('data.xlsx')
ds_pr = pd.read_excel('2022-2023.xlsx')
dataset.sort_values(by='month', ascending=False)
ds_pr.sort_values(by='month', ascending=False)
x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values
x_2022_2023 = ds_pr.iloc[:, 1:5].values
#Создаём и обучаем модель линейной регрессии
lnr = LinearRegression()
lnr.fit(x, y.reshape(-1, 1))
#Создаём и обучаем модель полиномиальной регрессии
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
lr = LinearRegression()
lr.fit(xp,y)
xp_2022_2023 = pf.fit_transform(x_2022_2023)
#Строим графики линейной регрессии
zlnr = round(lnr.score(x, y.reshape(-1, 1)), 2)
plt.scatter(x[:, 3], y, color='red')
plt.plot(x_2022_2023[:,3], lnr.predict(x_2022_2023), linestyle='-', marker='s', color='g', markerfacecolor='#5aff44')
plt.plot(x[:,3], lnr.predict(x),linestyle='-', marker='o', color='k', markerfacecolor='#ff22aa')
plt.legend(['Будущие расходы', f'Качество модели {zlnr}', 'Текущие расходы'])
plt.title('Linear Regression')
plt.xlabel('MONTH')
plt.ylabel('RENT')
plt.show()
#Строим графики полиномиальной регрессии
zlr = round(lr.score(xp, y), 2)
plt.scatter(x[:, 3], y, color='red')
plt.plot(x[:, 3], lr.predict(xp), linestyle='-', marker='o', color='k', markerfacecolor='#ff22aa')
plt.plot(x_2022_2023[:, 3], lr.predict(xp_2022_2023),linestyle='-', marker='s', color='g', markerfacecolor='#5aff44')
plt.legend(['Будущие расходы', f'Качество модели {zlr}', 'Текущие расходы'])
plt.title('Polynomial  Regression')
plt.xlabel('MONTH')
plt.ylabel('RENT')
plt.show()