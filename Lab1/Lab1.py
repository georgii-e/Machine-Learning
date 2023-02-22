import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.precision', 2)

initial_df = pd.read_excel(r'F:\Egor\Уроки\Машинне навчання\Лаб1\вартість товарів — копия.xlsx', index_col=[0, 1])
df = initial_df.copy()
df.replace('—', np.nan, inplace=True)
# print(df)

df.sort_values(by="Грудень 2021", inplace=True)
mean = df.mean()
print("\nMean:\n", mean)
median = df.median()
print("\nMedian:\n", median)
mode = df["Січень"].mode()  # coz another columns have 54 modes
print("\nMode:\n", mode)
variance = df.var()
print("\nVariance:\n", variance)
RMSD = np.sqrt(variance)
print("\nRoot mean square deviation:\n", RMSD)
df.iloc[0:-1:8].plot(kind='bar')
plt.show()
#

print("Description of dataframe by months", df.describe(), end="\n\n")
print("The cost of all goods in December:", df["Грудень"], end="\n\n")
print("Beet price in december:", df["Грудень"].loc["Буряк", "кг"], end="\n\n")
print("Prices that are more than 200 or less than 50 in February and March:",
      df[["Лютий", "Березень"]][(df > 200) | (df < 50)], end="\n\n")
print("Millet price in april:", df.at[("Пшоно", "кг"), "Квітень"], end="\n\n")
print("Description of dataframe by products\n", df.T.describe(), end="\n\n")

# initial_df = pd.read_csv(r'F:\Egor\Уроки\Машинне навчання\Лаб1\TitanicSurvival.csv')
# df = initial_df.copy()
# df.columns = ['name', 'survived', 'sex', 'age', 'class']
# print(df.head(), end="\n\n")
# print(df.tail(), end="\n\n")
# print("Youngest passenger:\n", df.iloc[df['age'].idxmin()], end="\n\n")
# print("Oldest passenger:\n", df.iloc[df['age'].idxmax()], end="\n\n")
# print("Average age:", df['age'].mean().round(2), end="\n\n")
# print("Statistics on passengers who survived:\n", df[df['survived'] == 'yes'].describe(), end="\n\n")
# women_first_class = df[(df['class'] == '1st') & (df['sex'] == 'female')].copy()
# women_first_class.reset_index(drop=True, inplace=True)
# women_first_class.sort_values(by='age', inplace=True)
# print("Youngest woman from the 1st class:\n", women_first_class.iloc[women_first_class['age'].idxmin()], end="\n\n")
# print("Oldest woman from the 1st class:\n", women_first_class.iloc[women_first_class['age'].idxmax()], end="\n\n")
# print("Total female survivors amount:", women_first_class.loc[women_first_class['survived'] == 'yes'].shape[0])
# df.hist(bins=80)
# plt.show()
