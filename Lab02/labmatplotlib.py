import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-white.csv', sep=';')
# Get number of rows and columns
print(df.shape)

dflq = df[df['quality'] == 3]
dfhq = df[df['quality'] == 9]
print(dflq['alcohol'].describe())
print(dfhq['alcohol'].describe())

df_ordenado = df.sort_values(by='alcohol', ascending=False)
print(df_ordenado.head())
print(df_ordenado.tail())

df.dropna(inplace=True)
df.fillna(0, inplace=True)
df.interpolate(inplace=True)

grupo = df.groupby('quality')
print(grupo['alcohol'].mean())
