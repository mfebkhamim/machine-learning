# Import modul atau libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Membaca data
df = pd.read_csv(r"F:\Portofolio\Belajar Python\1. Linear Regression\Mencoba Python\USA_Housing.csv")
df

# Menampilkan informasi data
df.head()
df.info()
df.describe()

# Exploratory Data Analysis

# Pairplot
sns.pairplot(df)

# Heatmap untuk cek korelasi data
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.tight_layout()

# PELATIHAN DATA
from sklearn.model_selection import train_test_split

# Pendefinisian X dan y
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# Mekanisme pelatihan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Training model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

# Mengecek parameter hasil pelatihan
print('Intercept:', lm.intercept_)
print('Coefficients:', lm.coef_)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']) # Untuk cek nama masing-masing kolom dan nilainya

# Mencari seberapa jauh hasil prediksi dengan data sebenarnya
prediksi = lm.predict(X_test)
plt.scatter(y_test, prediksi)

# Metrik evaluasi untuk Regresi Linear
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediksi))
print('MSE:', metrics.mean_squared_error(y_test, prediksi))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediksi)))