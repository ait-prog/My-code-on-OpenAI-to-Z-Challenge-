import folium 
import pandas as pd
import numpy as np
import re
from pyproj import Transformer
import openpyxl
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


arch_df = pd.read_csv('/kaggle/input/archaeological-survey-data/submit.csv', na_values=["", " ", "NA", "nan"])
amazon_df = pd.read_csv('/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv', na_values=["", " ", "NA", "nan"])
science_df = pd.read_csv('/kaggle/input/science-data/science.ade2541_data_s2.csv', na_values=["", " ", "NA", "nan"])
amazon_df.columns = amazon_df.columns.str.strip() 

# Заполнение пропущенных значений
arch_df.fillna("Unknown", inplace=True)
amazon_df.fillna({"latitude": 0, "longitude": 0, "mountain_range": "Unknown"}, inplace=True)
science_df.fillna(method='ffill', inplace=True)

# Вывод первых 10 строк
print("ARCHAEOLOGICAL DATA:")
print(arch_df.head(10), "\n")

print("AMAZON GEOGLYPHS:")
print(amazon_df.head(10), "\n")

print("SCIENCE REMOTE SENSING:")
print(science_df.head(10), "\n")

# Визуализация по типу объектов (если такой столбец есть)
if 'feature_type' in amazon_df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=amazon_df, x='feature_type')
    plt.title('Распределение геоглифов по типу (feature_type)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Столбец 'feature_type' не найден в amazon_df. Проверь названия колонок:")
    print(amazon_df.columns.tolist())
