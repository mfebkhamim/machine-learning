# Import libraries yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
%matplotlib inline

# Load dataset
yelp = pd.read_csv(r'C:\Users\Provicom\Downloads\Khamim\Karier\20-Natural-Language-Processing\yelp.csv')

# Cek informasi dataset
yelp.head()
yelp.info()
yelp.describe()

# Menambahkan kolom 'text length' untuk menghitung panjang teks review
yelp['text length'] = yelp['text'].apply(len)

# Exploratory Data Analysis (EDA)
FacetGrid = sns.FacetGrid(yelp, col='stars')
FacetGrid.map(plt.hist, 'text length', bins=30)

# Mengecek plot untuk cek outlier
boxplot = sns.boxplot(x='stars', y='text length', data=yelp)

# Mengecek jumlah data tiap label
countplot = sns.countplot(x='stars', data=yelp, palette='viridis')

# Melihat korelasi antarfitur 
sns.heatmap(yelp.groupby('stars').mean(numeric_only=True).corr(), annot=True, cmap='coolwarm')

# Preprocessing data
X = yelp_class['text']
y = yelp_class['stars']

# Melakukan CountVectorizer untuk mengubah teks menjadi fitur numerik
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
X = CV.fit_transform(X)

# Split data menjadi training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Melatih model Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Melakukan prediksi
predictions = nb.predict(X_test)

# Melakukan  evaluasi model
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediksi))
print(classification_report(y_test, prediksi))

# Coba menggunakan TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF = TfidfVectorizer()

# Menggabungkan pemrosesan data dengan pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # Bag of Words
    ('tfidf', TfidfTransformer()),  # Term Frequency-Inverse Document Frequency
    ('classifier', MultinomialNB()),  # Naive Bayes Classifier
])

# Melatih model dengan pipeline
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train, y_train)

# Menghasilkan prediksi dengan pipeline
predictions = pipeline.predict(X_test)

# Melakukan evaluasi model dengan pipeline
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))