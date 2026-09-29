import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / 'car_fuel_efficiency_2026.csv'

df = pd.read_csv(csv_path)

c = len(df)

print(f'Total of rows: {c}')
###############################################################################
fuel_type = df.fuel_type.nunique()

print(f"Fuel types: {fuel_type}")
###############################################################################
miss_values = df.isna().sum()

print(f"Number of missing values: {miss_values}")
###############################################################################

fe = df['fuel_efficiency_mpg'].max()
print(f"Max value of fuel_efficiency: {fe}")
###############################################################################
m_h = df['horsepower'].median()
freq_v = df['horsepower'].value_counts().index[0]
fillV = df['horsepower'].fillna(freq_v)
mean_h_again  = df['horsepower'].median()

print(f"Median value of horsepower: {m_h}") 
print(f"Most frecuency value: {freq_v}") 
print(f"Fill value of horsepower: {fillV}") 
print(f"Median value of horsepower: {mean_h_again}") 
###############################################################################
import numpy as np

asia = df[df.origin == 'Asia']

asia = asia[['vehicle_weight', 'model_year']]

asia = asia.head(7)

X = asia.values

XTX = X.T @ X

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print(f"w: {w}")
print(f"Sum of all elements of w: {w.sum()}")