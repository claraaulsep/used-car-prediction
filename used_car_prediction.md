---
jupyter:
  kernelspec:
    display_name: semesterlima
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.16
  nbformat: 4
  nbformat_minor: 5
---

::: {#b56893a0 .cell .code execution_count="2"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

```
:::

::: {#df6eec9f .cell .code execution_count="3"}
``` python
df = pd.read_csv('used_car_prediction.csv')
```
:::

::: {#0121b5a5 .cell .code execution_count="4"}
``` python
df.tail()
```

::: {.output .execute_result execution_count="4"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make_year</th>
      <th>mileage_kmpl</th>
      <th>engine_cc</th>
      <th>fuel_type</th>
      <th>owner_count</th>
      <th>price_usd</th>
      <th>brand</th>
      <th>transmission</th>
      <th>color</th>
      <th>service_history</th>
      <th>accidents_reported</th>
      <th>insurance_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9995</th>
      <td>1996</td>
      <td>15.25</td>
      <td>1500</td>
      <td>Petrol</td>
      <td>2</td>
      <td>5498.71</td>
      <td>Hyundai</td>
      <td>Manual</td>
      <td>Silver</td>
      <td>NaN</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>1997</td>
      <td>14.45</td>
      <td>5000</td>
      <td>Diesel</td>
      <td>1</td>
      <td>9588.31</td>
      <td>Honda</td>
      <td>Manual</td>
      <td>Gray</td>
      <td>Full</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>2002</td>
      <td>19.41</td>
      <td>1000</td>
      <td>Diesel</td>
      <td>4</td>
      <td>4557.10</td>
      <td>Kia</td>
      <td>Manual</td>
      <td>Black</td>
      <td>Full</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>2005</td>
      <td>12.95</td>
      <td>4000</td>
      <td>Petrol</td>
      <td>5</td>
      <td>7413.59</td>
      <td>BMW</td>
      <td>Automatic</td>
      <td>Blue</td>
      <td>Full</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>2007</td>
      <td>18.81</td>
      <td>5000</td>
      <td>Petrol</td>
      <td>1</td>
      <td>11634.09</td>
      <td>BMW</td>
      <td>Automatic</td>
      <td>Silver</td>
      <td>Full</td>
      <td>0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#67d4c5a4 .cell .code execution_count="5"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 12 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   make_year           10000 non-null  int64  
     1   mileage_kmpl        10000 non-null  float64
     2   engine_cc           10000 non-null  int64  
     3   fuel_type           10000 non-null  str    
     4   owner_count         10000 non-null  int64  
     5   price_usd           10000 non-null  float64
     6   brand               10000 non-null  str    
     7   transmission        10000 non-null  str    
     8   color               10000 non-null  str    
     9   service_history     7962 non-null   str    
     10  accidents_reported  10000 non-null  int64  
     11  insurance_valid     10000 non-null  str    
    dtypes: float64(2), int64(4), str(6)
    memory usage: 937.6 KB
:::
:::

::: {#e516940e .cell .code execution_count="6"}
``` python
df.shape
```

::: {.output .execute_result execution_count="6"}
    (10000, 12)
:::
:::

::: {#af2dd4d4 .cell .code execution_count="7"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="7"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make_year</th>
      <th>mileage_kmpl</th>
      <th>engine_cc</th>
      <th>owner_count</th>
      <th>price_usd</th>
      <th>accidents_reported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2009.206900</td>
      <td>17.960753</td>
      <td>2287.130000</td>
      <td>3.003500</td>
      <td>7179.754532</td>
      <td>0.492200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.373858</td>
      <td>5.025486</td>
      <td>1291.276927</td>
      <td>1.418904</td>
      <td>2795.270940</td>
      <td>0.694109</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1995.000000</td>
      <td>5.000000</td>
      <td>800.000000</td>
      <td>1.000000</td>
      <td>1000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2002.000000</td>
      <td>14.540000</td>
      <td>1200.000000</td>
      <td>2.000000</td>
      <td>5176.547500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>17.970000</td>
      <td>1800.000000</td>
      <td>3.000000</td>
      <td>6961.260000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>21.352500</td>
      <td>3000.000000</td>
      <td>4.000000</td>
      <td>8993.732500</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023.000000</td>
      <td>35.000000</td>
      <td>5000.000000</td>
      <td>5.000000</td>
      <td>17647.630000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#d32cec8f .cell .code execution_count="8"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="8"}
    make_year                0
    mileage_kmpl             0
    engine_cc                0
    fuel_type                0
    owner_count              0
    price_usd                0
    brand                    0
    transmission             0
    color                    0
    service_history       2038
    accidents_reported       0
    insurance_valid          0
    dtype: int64
:::
:::

::: {#9602a93e .cell .code execution_count="9"}
``` python
df.duplicated().sum()
```

::: {.output .execute_result execution_count="9"}
    np.int64(0)
:::
:::

::: {#a12aa284 .cell .code execution_count="10"}
``` python
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='price_usd', kde=True)

plt.title('Distribution Harga Mobil')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()
```

::: {.output .display_data}
![](4f090967e76d6536b6dd748cc57df74cee983464.png)
:::
:::

::: {#362bcad8 .cell .code execution_count="11"}
``` python
df['make_year'].hist()
plt.title('Distribusi Tahun Pembuatan Mobil')
plt.xlabel('tahun pembuatan')
plt.ylabel('frekuensi')

plt.show()
```

::: {.output .display_data}
![](8b353bdffa04fbe3c67346d5da00aacf1002527d.png)
:::
:::

::: {#bc1a2f36 .cell .code execution_count="12"}
``` python
df['mileage_kmpl'].hist()
plt.title('Distribusi Efisiensi Bahan Bakar')
plt.xlabel('efisiensi bahan bakar (km/l)')
plt.ylabel('frekuensi')
plt.show()
```

::: {.output .display_data}
![](94d63f2995665faa73d680c10adcf959106f524b.png)
:::
:::

::: {#182a8e96 .cell .code execution_count="13"}
``` python
# 1. Tahun produksi vs harga
sns.scatterplot(data=df, x='make_year', y='price_usd')
plt.title('Tahun Produksi vs Harga Mobil')
plt.show()

# 2. Kapasitas mesin vs harga
sns.scatterplot(data=df, x='engine_cc', y='price_usd')
plt.title('Kapasitas Mesin vs Harga Mobil')
plt.show()

# 3. Efisiensi bahan bakar vs harga
sns.scatterplot(data=df, x='mileage_kmpl', y='price_usd')
plt.title('Efisiensi Bahan Bakar vs Harga Mobil')
plt.show()

# 4. Jumlah pemilik sebelumnya vs harga
sns.scatterplot(data=df, x='owner_count', y='price_usd')
plt.title('Jumlah Pemilik vs Harga Mobil')
plt.show()

# 9. Jumlah kecelakaan vs harga
sns.scatterplot(data=df, x='accidents_reported', y='price_usd')
plt.title('Jumlah Kecelakaan vs Harga Mobil')
plt.show()
```

::: {.output .display_data}
![](498abb2bc20d4e9ddf827b7d111454cd4c5059d7.png)
:::

::: {.output .display_data}
![](7348944cd3c326bddbede55444260f4a89a3a10e.png)
:::

::: {.output .display_data}
![](fe373f7e4c0cf98c8721494e44b4211a9831ab72.png)
:::

::: {.output .display_data}
![](036a7a2f9750790e4e8e6b9a0e14dc60d8ab97d2.png)
:::

::: {.output .display_data}
![](da2a5e54bd3c8451c31741f5e4a1c9b9b870e4b2.png)
:::
:::

::: {#056d30fb .cell .code execution_count="14"}
``` python
# 5. Merek mobil vs harga
sns.boxplot(data=df, x='brand', y='price_usd')
plt.title('Merek Mobil vs Harga')
plt.xticks(rotation=45)
plt.show()

# 6. Jenis bahan bakar vs harga
sns.boxplot(data=df, x='fuel_type', y='price_usd')
plt.title('Jenis Bahan Bakar vs Harga')
plt.show()

# 7. Jenis transmisi vs harga
sns.boxplot(data=df, x='transmission', y='price_usd')
plt.title('Transmisi vs Harga')
plt.show()

# 8. Riwayat servis vs harga
sns.boxplot(data=df, x='service_history', y='price_usd')
plt.title('Riwayat Servis vs Harga')
plt.show()

# 10. Status asuransi vs harga
sns.boxplot(data=df, x='insurance_valid', y='price_usd')
plt.title('Status Asuransi vs Harga')
plt.show()
```

::: {.output .display_data}
![](bd215d6cc3b1d41776789c87cc92ab1198cb8a84.png)
:::

::: {.output .display_data}
![](120a5b4dcbfda7ed6ae80ada90e74fdd42162921.png)
:::

::: {.output .display_data}
![](d8257ae02158be0b256f21c7c161c0a8c7e1ba98.png)
:::

::: {.output .display_data}
![](feace65ec0b944998590ea5fc06a2529159b49b9.png)
:::

::: {.output .display_data}
![](9c6eceb55caeab02dbf8aee04d9127ca257dfa55.png)
:::
:::

::: {#f708406f .cell .code execution_count="15"}
``` python
df.groupby('color')['price_usd'].agg(
    ['count', 'mean', 'median']
).sort_values('median', ascending=False)
```

::: {.output .execute_result execution_count="15"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>color</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Red</th>
      <td>1678</td>
      <td>7287.139678</td>
      <td>7133.540</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>1702</td>
      <td>7216.539318</td>
      <td>7027.635</td>
    </tr>
    <tr>
      <th>Gray</th>
      <td>1705</td>
      <td>7190.391296</td>
      <td>6971.680</td>
    </tr>
    <tr>
      <th>White</th>
      <td>1651</td>
      <td>7147.743137</td>
      <td>6915.310</td>
    </tr>
    <tr>
      <th>Blue</th>
      <td>1634</td>
      <td>7197.815367</td>
      <td>6898.645</td>
    </tr>
    <tr>
      <th>Silver</th>
      <td>1630</td>
      <td>7033.989957</td>
      <td>6803.635</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#b10f0693 .cell .markdown}
### Apakah mobil dengan kapasitas mesin besar dan tahun produksi lebih baru cenderung memiliki harga lebih tinggi?
:::

::: {#25b9b217 .cell .code execution_count="16"}
``` python
sns.scatterplot(
    data=df,
    x='make_year',
    y='price_usd',
    hue='engine_cc',
    palette='viridis'
)

plt.title('Tahun Produksi dan Kapasitas Mesin terhadap Harga')
plt.xlabel('Tahun Produksi')
plt.ylabel('Harga Mobil (USD)')
plt.legend(title='Kapasitas Mesin (CC)')
plt.show()
```

::: {.output .display_data}
![](a2ff544f8c654e97f31ab68fe9d5d446c868fd91.png)
:::
:::

::: {#ed841862 .cell .code execution_count="17"}
``` python
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.show()
```

::: {.output .display_data}
![](d46c17d2509bcd445ea8fcec9bcf6400cc06cb1d.png)
:::
:::

::: {#d24c636d .cell .code execution_count="18"}
``` python
df['service_history'] = df['service_history'].fillna('Unknown')
```
:::

::: {#bb3b799a .cell .code execution_count="19"}
``` python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['price_usd'])
y = df['price_usd']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
```

::: {.output .stream .stdout}
    (8000, 11)
    (2000, 11)
:::
:::

::: {#f78cfe13 .cell .markdown}
## Feature Engineering
:::

::: {#27b6339b .cell .code execution_count="20"}
``` python
X_train['car_age'] = 2026 - X_train['make_year']
X_test['car_age'] = 2026 - X_test['make_year']
```
:::

::: {#e39047f6 .cell .code execution_count="21"}
``` python
categorical_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
print(categorical_cols)
```

::: {.output .stream .stdout}
    ['fuel_type', 'brand', 'transmission', 'color', 'service_history', 'insurance_valid']
:::
:::

::: {#058ddb9c .cell .code execution_count="22"}
``` python
# Inisialisasi encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# FIT HANYA di X_train, lalu TRANSFORM ke X_train dan X_test
encoded_train = pd.DataFrame(
    encoder.fit_transform(X_train[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_train.index
)

encoded_test = pd.DataFrame(
    encoder.transform(X_test[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)
```
:::

::: {#18c0ef57 .cell .code execution_count="23"}
``` python
# Gabungkan fitur numerik lama dengan fitur hasil encoding
X_train = X_train.drop(columns=categorical_cols).join(encoded_train)
X_test = X_test.drop(columns=categorical_cols).join(encoded_test)
```
:::

::: {#7262d667 .cell .code execution_count="24"}
``` python
# LANGKAH 4: Verifikasi Hasil Akhir
# =========================================================
print("Bentuk X_train setelah pemrosesan:", X_train.shape)
print("Bentuk X_test setelah pemrosesan :", X_test.shape)
X_train.head()
```

::: {.output .stream .stdout}
    Bentuk X_train setelah pemrosesan: (8000, 32)
    Bentuk X_test setelah pemrosesan : (2000, 32)
:::

::: {.output .execute_result execution_count="24"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make_year</th>
      <th>mileage_kmpl</th>
      <th>engine_cc</th>
      <th>owner_count</th>
      <th>accidents_reported</th>
      <th>car_age</th>
      <th>fuel_type_Diesel</th>
      <th>fuel_type_Electric</th>
      <th>fuel_type_Petrol</th>
      <th>brand_BMW</th>
      <th>...</th>
      <th>color_Blue</th>
      <th>color_Gray</th>
      <th>color_Red</th>
      <th>color_Silver</th>
      <th>color_White</th>
      <th>service_history_Full</th>
      <th>service_history_Partial</th>
      <th>service_history_Unknown</th>
      <th>insurance_valid_No</th>
      <th>insurance_valid_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9254</th>
      <td>2002</td>
      <td>9.70</td>
      <td>1800</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1561</th>
      <td>2004</td>
      <td>16.10</td>
      <td>1500</td>
      <td>3</td>
      <td>0</td>
      <td>22</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1670</th>
      <td>2005</td>
      <td>10.51</td>
      <td>5000</td>
      <td>2</td>
      <td>0</td>
      <td>21</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6087</th>
      <td>2019</td>
      <td>14.27</td>
      <td>2000</td>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6669</th>
      <td>2015</td>
      <td>21.72</td>
      <td>1000</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>
```
:::
:::

::: {#af9a531c .cell .code execution_count="25"}
``` python
X_train.head(5)
```

::: {.output .execute_result execution_count="25"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>make_year</th>
      <th>mileage_kmpl</th>
      <th>engine_cc</th>
      <th>owner_count</th>
      <th>accidents_reported</th>
      <th>car_age</th>
      <th>fuel_type_Diesel</th>
      <th>fuel_type_Electric</th>
      <th>fuel_type_Petrol</th>
      <th>brand_BMW</th>
      <th>...</th>
      <th>color_Blue</th>
      <th>color_Gray</th>
      <th>color_Red</th>
      <th>color_Silver</th>
      <th>color_White</th>
      <th>service_history_Full</th>
      <th>service_history_Partial</th>
      <th>service_history_Unknown</th>
      <th>insurance_valid_No</th>
      <th>insurance_valid_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9254</th>
      <td>2002</td>
      <td>9.70</td>
      <td>1800</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1561</th>
      <td>2004</td>
      <td>16.10</td>
      <td>1500</td>
      <td>3</td>
      <td>0</td>
      <td>22</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1670</th>
      <td>2005</td>
      <td>10.51</td>
      <td>5000</td>
      <td>2</td>
      <td>0</td>
      <td>21</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6087</th>
      <td>2019</td>
      <td>14.27</td>
      <td>2000</td>
      <td>5</td>
      <td>1</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6669</th>
      <td>2015</td>
      <td>21.72</td>
      <td>1000</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>
```
:::
:::

::: {#95dfcbf5 .cell .code execution_count="26"}
``` python
print("Jumlah kolom sebelum encoding:", df.shape[1])
print("Jumlah kolom setelah encoding:", X_train.shape[1])
```

::: {.output .stream .stdout}
    Jumlah kolom sebelum encoding: 12
    Jumlah kolom setelah encoding: 32
:::
:::

::: {#54ea7841 .cell .code execution_count="27"}
``` python
#supaya tidak redundan dan mengurangi multikolinearitas, kita bisa menghapus kolom 'make_year' karena kita sudah membuat kolom baru 'car_age' yang merepresentasikan informasi yang sama.
X_train = X_train.drop(columns=['make_year'])
X_test = X_test.drop(columns=['make_year'])

```
:::

::: {#e03b4104 .cell .code execution_count="32"}
``` python
scaler = StandardScaler()

# Fit & Transform HANYA pada data train
X_train_scaled = scaler.fit_transform(X_train)

# Transform HANYA pada data test (tanpa fit)
X_test_scaled = scaler.transform(X_test)

# Optional: Kembalikan ke DataFrame agar nama kolom tidak hilang
X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    X_test_scaled, columns=X_test.columns, index=X_test.index
)
```
:::

::: {#2eedb98c .cell .markdown}
## Modeling
:::

::: {#c4a097c7 .cell .code execution_count="33"}
``` python
# 1. Inisialisasi daftar model yang akan dibandingkan
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}
```
:::

::: {#1bd24886 .cell .code execution_count="34"}
``` python
results = []

for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Hitung metrik
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results.append({
        'Model': name,
        'R2 Train': round(r2_tr, 4),
        'R2 Test': round(r2_te, 4),
        'MAE ($)': round(mae, 2),
        'RMSE ($)': round(rmse, 2)
    })
```
:::

::: {#4c9cb0d9 .cell .code execution_count="35"}
``` python
df_results = pd.DataFrame(results).sort_values(by='R2 Test', ascending=False).reset_index(drop=True)
display(df_results)
```

::: {.output .display_data}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>R2 Train</th>
      <th>R2 Test</th>
      <th>MAE ($)</th>
      <th>RMSE ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>0.8709</td>
      <td>0.8765</td>
      <td>791.00</td>
      <td>991.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gradient Boosting</td>
      <td>0.8780</td>
      <td>0.8712</td>
      <td>805.09</td>
      <td>1012.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.9788</td>
      <td>0.8520</td>
      <td>864.42</td>
      <td>1085.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree</td>
      <td>1.0000</td>
      <td>0.7032</td>
      <td>1226.11</td>
      <td>1536.75</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#16f6539c .cell .code execution_count="36"}
``` python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R2 Score (Test Set)
sns.barplot(data=df_results, x='R2 Test', y='Model', ax=axes[0], palette='Blues_r')
axes[0].set_title('Perbandingan R² Score (Mendekati 1.0 = Terbaik)', fontweight='bold')
axes[0].set_xlim(0, 1.0)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_width():.4f}", (p.get_width() - 0.1, p.get_y() + p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

# Plot 2: MAE (Test Set)
sns.barplot(data=df_results, x='MAE ($)', y='Model', ax=axes[1], palette='Reds_r')
axes[1].set_title('Perbandingan MAE (Semakin Kecil = Terbaik)', fontweight='bold')
for p in axes[1].patches:
    axes[1].annotate(f"${p.get_width():,.2f}", (p.get_width() / 2, p.get_y() + p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
```

::: {.output .stream .stderr}
    C:\Users\User\AppData\Local\Temp\ipykernel_22920\2632622150.py:4: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

      sns.barplot(data=df_results, x='R2 Test', y='Model', ax=axes[0], palette='Blues_r')
    C:\Users\User\AppData\Local\Temp\ipykernel_22920\2632622150.py:12: FutureWarning: 

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

      sns.barplot(data=df_results, x='MAE ($)', y='Model', ax=axes[1], palette='Reds_r')
:::

::: {.output .display_data}
![](c181149f5dffb4a11392bdc0d365cd8414c8771e.png)
:::
:::

::: {#2a799fed .cell .code execution_count="38"}
``` python
# Ambil 10 fitur yang paling dominan mempengaruhi harga
importances = pd.Series(model_rf.feature_importances_, index=X_train.columns)

plt.figure(figsize=(10, 5))
importances.nlargest(10).plot(kind='barh', color='#2b5c8f')
plt.title('10 Fitur Paling Berpengaruh Terhadap Harga Mobil', fontsize=12, fontweight='bold')
plt.xlabel('Tingkat Kepentingan (Importance Score)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](159f1f501851468c870ae57b34126ab9b2416846.png)
:::
:::

::: {#7a804bb3 .cell .code execution_count="37"}
``` python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.4, color='royalblue', label='Data Prediksi')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal (Perfect Prediction)')

plt.xlabel('Harga Asli (Actual price_usd)', fontsize=11)
plt.ylabel('Harga Prediksi (Predicted price_usd)', fontsize=11)
plt.title('Evaluasi Visual: Harga Asli vs Harga Prediksi', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](778073315460b9fc0209ee80a1d73ab349893728.png)
:::
:::

::: {#d1ad4784 .cell .code}
``` python

sample_data = X_test.iloc[[0]]
actual_price
predicted_price = models['Linear Regression'].predict(sample_data)[0]

print(f"Harga Asli Mobil     : ${actual_price:,.2f}")
print(f"Harga Hasil Prediksi : ${predicted_price:,.2f}")
print(f"Selisih (Error)      : ${abs(actual_price - predicted_price):,.2f}")
```

::: {.output .stream .stdout}
    Harga Asli Mobil     : $12,772.12
    Harga Hasil Prediksi : $12,124.39
    Selisih (Error)      : $647.73
:::
:::
