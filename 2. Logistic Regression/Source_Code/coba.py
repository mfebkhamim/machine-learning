# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplorlib inline


# PRA-PEMROSESAN DATA

# Membaca data
data = pd.read_csv(r'E:\Perkuliahan\Karier\13-Logistic-Regression\advertising.csv')

# Menampilkan Informasi Dasar tentang Data
data.head()
data.info()
data.describe()

# EXPLORATORY DATA ANALYSIS (EDA)

# Membuat Histogram untuk Age
sns.set_style('whitegrid')
data['Age'].hist(bins=30)
plt.xlabel('Age')

# Membuat Jointplot antara Age dan Area Income
sns.jointplot(x='Age', y='Area Income', data=data)

# Membuat Jointplot antara Daily Time Spent on Site dan Age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')

# Membuat jointplot antara Daily Time Spent on Site dan Daily Internet Usage
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

# Membuat pairplot dengan hue berdasarkan Clicked on Ad
# Clicked on Ad adalah variabel target
sns.pairplot(data,hue='Clicked on Ad',palette='bwr')

# PELATIHAN MODEL LOGISTIC REGRESSION

# Memisahkan data train dan data test
from sklearn.model_selection import train_test_split
X = data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Melatih model Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# MELAKUKAN PREDIKSI
predictions = logmodel.predict(X_test)

# Evaluasi Model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))