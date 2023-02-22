import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.precision', 2)
plt.style.use('seaborn-v0_8')

initial_df = pd.read_csv(r'F:\Egor\Уроки\Машинне навчання\Лаб2\1895-2018.csv')
df = initial_df.copy()
df.columns = ['Date', 'Temperature', 'Anomaly']
df.drop(index=range(4), inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.astype({'Date': 'int64', 'Temperature': 'float'})
df.loc[:, 'Temperature'] = df.loc[:, 'Temperature'].apply(lambda x: 5 / 9 * (x - 32))
df.loc[:, 'Date'] = df.loc[:, 'Date'].floordiv(100)
print(df.head())
dates = df.loc[:, 'Date'].values
temps = df.loc[:, 'Temperature'].values
linear_regression = stats.linregress(dates, temps)

plt.plot(dates, linear_regression.slope * dates + linear_regression.intercept)
plt.show()


def temperature_forecast(regression, years):
    for year in years:
        print(f'Projected temperature of {year} year: {regression.slope * year + regression.intercept:.2f}')
    print()


temperature_forecast(linear_regression, range(2019, 2024))
temperature_forecast(linear_regression, range(1890, 1895))

sns.regplot(x=dates, y=temps)
plt.xlabel('Date, years')
plt.ylabel('Temperature, celsius')
plt.xlim(1895, 2018)
plt.ylim(-10, 10)
plt.show()

