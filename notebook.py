# %% [markdown]
# # Books Recommender System
#
# Industri buku telah mengalami pertumbuhan yang signifikan dalam beberapa tahun terakhir, dengan ribuan judul baru diterbitkan setiap tahunnya. Di Indonesia, data dari Ikatan Penerbit Indonesia (IKAPI) menunjukkan bahwa lebih dari 30.000 judul buku diterbitkan setiap tahun [[1]](https://kumparan.com/kumparannews/industri-buku-di-indonesia/full). Namun, rata-rata masyarakat Indonesia hanya membeli dua buku per tahun, yang menunjukkan tantangan dalam meningkatkan minat baca.
#
# Selain itu, data dari Perpustakaan Nasional (Perpusnas) menunjukkan bahwa pada tahun 2021, hampir 160.000 ISBN diterbitkan, dengan 147.000 di antaranya adalah buku. Namun, jumlah ini menurun menjadi 107.800 ISBN pada tahun 2022 dan sedikit meningkat menjadi 108.000 ISBN pada tahun 2023. Hingga pertengahan 2024, sudah ada 70.000 ISBN yang diterbitkan, dengan 65.000 di antaranya untuk buku [[2]](https://data.goodstats.id/statistic/jumlah-isbn-indonesia-turun-sejak-2021-jZosj). Dengan semakin banyaknya buku yang diterbitkan setiap tahun, pengguna sering mengalami kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka, khususnya buku digital yang kian menjadi metode berbeda untuk membaca dan menemukan buku baru [[3]](https://www.researchgate.net/publication/383038644_Sistem_Rekomendasi_Buku_Berbasis_Konten_Menggunakan_Metode_Collaborative_Filtering). Hal ini menekankan pentingnya sistem rekomendasi buku yang efektif untuk membantu pembaca menavigasi pilihan yang luas dan menemukan bacaan yang sesuai dengan minat mereka.
#
# Data berdasarkan penelitian menunjukkan bahwa penggunaan sistem rekomendasi berbasis content-based filtering dapat meningkatkan user engagement dan membantu pengguna dalam menemukan buku yang relevan dengan preferensi mereka [[4]](https://www.researchgate.net/publication/348968927_Personalized_Book_Recommendation_System_using_Machine_Learning_Algorithm). Sejalan dengan ini, studi lain mengonfirmasi bahwa algoritma machine learning berbasis collaborative filtering dapat meningkatkan akurasi rekomendasi buku secara signifikan [[5]](https://ieeexplore.ieee.org/document/7019651).
#
# Sistem rekomendasi buku menjadi sangat penting karena beberapa alasan:
# 1. Membantu pengguna menemukan konten yang relevan di antara ribuan pilihan yang ada.
# 2. Meningkatkan engagement user pada platform penjualan atau penyedia layanan buku.
# 3. Memberikan pengalaman personalisasi yang lebih baik kepada pembaca.
# 4. Membantu penerbit dan distributor buku dalam memasarkan konten mereka secara lebih efektif.
#
# **Referensi Riset Terkait:**
#
# [1] [Industri Buku di Indonesia](https://kumparan.com/kumparannews/industri-buku-di-indonesia/full)
# [2] [Jumlah ISBN Indonesia Turun Sejak 2021](https://data.goodstats.id/statistic/jumlah-isbn-indonesia-turun-sejak-2021-jZosj)
# [3] [Sistem Rekomendasi Buku Berbasis Konten Menggunakan Metode Collaborative Filtering](https://www.researchgate.net/publication/383038644_Sistem_Rekomendasi_Buku_Berbasis_Konten_Menggunakan_Metode_Collaborative_Filtering)
# [4] [Personalized Book Recommendation System using Machine Learning Algorithm](https://www.researchgate.net/publication/348968927_Personalized_Book_Recommendation_System_using_Machine_Learning_Algorithm)
# [5] [Book recommendation system based on collaborative filtering and association rule mining for college students](https://ieeexplore.ieee.org/document/7019651)

# %% [markdown]
# ## Data Understanding
#
# Dataset yang digunakan adalah [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data) dari Kaggle. Dataset ini terdiri dari tiga file yaitu books.csv, ratings.csv, dan users.csv, berikut adalah informasi mengenai dataset:
#
# ### Informasi Dataset:
# - Jumlah Data:
#   - books.csv: 271.360 rows, 8 columns
#   - ratings.csv: 1.149.780 rows, 3 columns
#   - users.csv: 278.858 rows, 3 columns
#
# ### Variabel-variabel pada Dataset:
#
# 1. books.csv (271.360 entries):
#    - ISBN: ID unik untuk setiap buku
#    - Book-Title: Judul buku
#    - Book-Author: Penulis buku
#    - Year-Of-Publication: Tahun terbit buku
#    - Publisher: Penerbit buku
#    - Image-URL-S: URL gambar sampul buku
#    - Image-URL-M: URL gambar sampul buku
#    - Image-URL-L: URL gambar sampul buku
#
# 2. ratings.csv (1.149.780 users):
#    - User-ID: ID unik untuk setiap user
#    - ISBN: ID buku yang diberi rating
#    - Book-Rating: Rating yang diberikan (skala 0-10)
#
# 3. users.csv (278.858 users):
#    - User-ID: ID unik untuk setiap user
#    - Location: Lokasi pengguna
#    - Age: Usia pengguna

# %% [markdown]
# ## Data Preparation

# %%
import os
from google.colab import drive
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers, models

# %% [markdown]
# Mount google drive untuk menyimpan dataset yang didownload dari kaggle

# %%
# Mount Google Drive
drive.mount('/content/drive')

# %% [markdown]
# Persiapkan Kaggle dengan melakukan instalasi dan konfigurasi kaggle.json

# %%
# Prepare Kaggle
!pip install kaggle
!mkdir ~/.kaggle
!cp / content/drive/MyDrive/Kaggle/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# %% [markdown]
# Download dataset book-recommendations-dataset

# %%
# Download dataset
!kaggle datasets download - d arashnic/book-recommendation-dataset - p / content/drive/MyDrive/Kaggle/Dataset

# %% [markdown]
# Unzip dataset yang telah di-download

# %%
# Unzip dataset
!unzip / content/drive/MyDrive/Kaggle/Dataset/book-recommendation-dataset.zip - d / content/drive/MyDrive/Kaggle/Dataset

# %% [markdown]
# Selanjutnya mari kita baca file Books.csv, rating.csv dan users.csv, lalu lihat berapa data yang ada dalam file tersebut. Data books.csv memiliki terlalu banyak data yang dapat membuat crash pada notebook, maka kita akan menggunakan sebagian data saja dengan mengambil 85.000 data acak dan 5 kolom saja.

# %%
cols = ['ISBN', 'Book-Title', 'Book-Author',
        'Year-Of-Publication', 'Publisher']
books = pd.read_csv(
    '/content/drive/MyDrive/Kaggle/Dataset/Books.csv', usecols=cols)
ratings = pd.read_csv('/content/drive/MyDrive/Kaggle/Dataset/Ratings.csv')
users = pd.read_csv('/content/drive/MyDrive/Kaggle/Dataset/Users.csv')

print(f'Books: {books.shape}')
print(f'Rating: {ratings.shape}')
print(f'Users: {users.shape}')

# %% [markdown]
# File Books.csv memiliki 271.360 baris dan 5 kolom, file ratings.csv memiliki 1.149.780 baris dan 3 kolom, dan file users.csv memiliki 278.858 baris dan 3 kolom.

# %% [markdown]
# ## Univariate Exploratory Data Analysis

# %% [markdown]
# Mari kita telusuri lebih dalam 3 file tersebut untuk memahami lebih lanjut apa yang disediakan dataset dan bagaimana kondisinya

# %% [markdown]
# ### Books.csv

# %%
books.info()

# %%
books.describe()

# %% [markdown]
# Hasil ini menunjukan bahwa terdapat 271.360 buku yang terdaftar dalam dataset dengan top author adalah `Agatha Christie` dan Publisher `Harlequin`. Berikut adalah 5 data pertama dari file books.csv:

# %%
books.head()

# %%
books.isnull().sum()

# %% [markdown]
# Terdapat 4 buah missing values yaitu 2 pada kolom `Book-Author` dan 2 pada kolom `Publisher`.

# %%
books.duplicated().sum()

# %% [markdown]
# Tidak ada data duplikat pada file books.csv

# %%
# Check for unique values
print(books['Year-Of-Publication'].unique())

# %% [markdown]
# Berdasarkan hasil di atas, kita dapat melihat bahwa kolom `Year-Of-Publication` memiliki data yang tidak konsisten, seperti `0` dan `DK Publishing Inc`. Kita akan melakukan cleaning data pada kolom tersebut.

# %%
books['Year-Of-Publication'][books['Year-Of-Publication'] == '0']

# %% [markdown]
# Terdapat 1048 baris yang memiliki nilai `0` pada kolom `Year-Of-Publication`, kita akan menghapusnya nanti.

# %%
books['Year-Of-Publication'].value_counts()

# %% [markdown]
# Untuk memahami lebih dalam terkait file books.csv mari kita lihat distribusi data dengan melakukan visualisasi data. Agar kolom `Year-Of-Publication` dapat divisualisasikan, kita akan melakukan cleaning data terlebih dahulu dengan membuat copy dari dataframes books.csv.

# %%
# buat copy agar dataframe utama tidak berubah
books_eda = books.copy()

# Menghapus data yang memiliki year of publication tidak valid
books_eda["Year-Of-Publication"] = pd.to_numeric(
    books["Year-Of-Publication"], errors="coerce")
books_eda = books_eda.dropna(subset=["Year-Of-Publication"])
books_eda["Year-Of-Publication"] = books_eda["Year-Of-Publication"].astype(int)


# %%
plt.figure(figsize=(10, 5))
sns.histplot(books_eda['Year-Of-Publication'], bins=50, kde=True)
plt.xlabel("Year of Publication")
plt.ylabel("Count")
plt.title("Distribution of Book Publication Years")
plt.show()

# %% [markdown]
# Distribusi data pada kolom `Year-Of-Publication` menunjukan rentang tahun yang jauh sekali, dengan konsentrasi data pada tahun 2000an.

# %%
# Top 10 Publisher
top_publishers = books_eda['Publisher'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_publishers.values, y=top_publishers.index,
            palette="viridis", hue=top_publishers.index)
plt.xlabel("Number of Books")
plt.ylabel("Publisher")
plt.title("Top 10 Publishers by Number of Books")

for i, v in enumerate(top_publishers.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()

# %% [markdown]
# Berdasarkan visualisasi data di atas, kita dapat melihat bahwa publisher `Harlequin` adalah publisher yang paling banyak menerbitkan buku dengan jumlah 7535 buku.

# %%
# Top 10 Author
top_authors = books_eda['Book-Author'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_authors.values, y=top_authors.index,
            palette="viridis", hue=top_authors.index)
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.title("Top 10 Authors by Number of Books")

for i, v in enumerate(top_authors.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()


# %% [markdown]
# Visualisasi data di atas menunjukkan bahwa author `Agatha Christie` adalah author yang paling banyak menerbitkan buku dengan jumlah 632 buku.

# %% [markdown]
# ### Ratings.CSV
#
# Selanjutnya mari kita eksplorasi file ratings.csv untuk memahami lebih dalam dataset tersebut

# %%
ratings.info()

# %% [markdown]
# File rating.csv memiliki size 26.3 MB dan terdapat 1.149.780 data rating buku yang diberikan oleh user.

# %%
ratings.describe()

# %%
ratings.isnull().sum()

# %% [markdown]
# Tidak terdapat missing values pada file ratings.csv

# %%
ratings.duplicated().sum()

# %% [markdown]
# Tidak terdapat data duplikat pada file ratings.csv

# %%
ratings.head()

# %%
plt.figure(figsize=(10, 8))
ax = sns.countplot(x=ratings["Book-Rating"],
                   palette="coolwarm", hue=ratings["Book-Rating"])
plt.xlabel("Book Rating")
plt.ylabel("Count")
plt.title("Distribution of Book Ratings")
plt.xticks(range(11))

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# %% [markdown]
# Dari visualisasi tersebut terdapat 716109 data yang memiliki rating 0, yang mungkin menunjukkan bahwa user tersebut belum memberikan rating pada buku tersebut. Data ini tidak relevan untuk digunakan dalam sistem rekomendasi, maka kita akan menghapus data tersebut. Di sisi lain rating 8 adalah rating yang paling banyak diberikan oleh user dan rating 1 adalah rating yang paling sedikit diberikan oleh user.

# %% [markdown]
# ### Users.CSV
#
# Selanjutnya mari kita eksplorasi file users.csv untuk memahami lebih dalam dataset tersebut

# %%
users.info()

# %%
users.describe()

# %% [markdown]
# Berdasarkan statistik deskriptif di atas, kita dapat melihat bahwa usia pengguna memiliki rentang nilai dari 0 hingga 244. Kita akan melakukan cleaning data pada kolom `Age` dengan menghapus data yang tidak relevan.

# %%
users.isnull().sum()

# %% [markdown]
# Data pada kolom `Age` memiliki missing values sebanyak 110.762 data, kita akan melakukan imputasi data pada kolom tersebut dengan menggunakan median.

# %%
users.duplicated().sum()

# %% [markdown]
# Tidak terdapat data duplikat pada file users.csv

# %%
users.head()

# %%
plt.figure(figsize=(10, 5))
sns.histplot(users["Age"].dropna(), bins=30, kde=True)
plt.xlabel("User Age")
plt.ylabel("Count")
plt.title("Distribution of User Ages")
plt.show()

# %% [markdown]
# Berdasarkan visualisasi data di atas, kita dapat melihat bahwa mayoritas pengguna berusia antara 20-30 tahun, dengan rentang data yang sangat jauh dari 0 hingga 244.

# %%
top_locations = users["Location"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_locations.values, y=top_locations.index,
            palette="magma", hue=top_locations.index)
plt.xlabel("Number of Users")
plt.ylabel("Location")
plt.title("Top 10 User Locations")

for i, v in enumerate(top_locations.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()

# %% [markdown]
# Top 10 lokasi pengguna yang paling banyak terdaftar dalam dataset adalah `london, england, united kingdom` dengan jumlah 2506 pengguna.

# %% [markdown]
# ## Data Preparation
#
# Sebelum melakukan modeling, data harus bersih dan rapi terlebih dahulu, maka dari itu kita harus melakukan pembersihan data dari duplikasi, dan missing values.

# %% [markdown]
# ### Books.csv
#
# Berdasarkan EDA yang telah dilakukan, terdapat beberapa missing values dalam data books.csv yaitu:
# - 2 missing values pada kolom `Book-Author`
# - 2 missing values pada kolom `Publisher`
#
# Selain itu, terdapat data yang tidak valid pada kolom `Year-Of-Publication` yaitu:
# - `DK Publishing Inc`
# - `Gallimard`
# - Tahun 0
# - Tahun 2037
#
# Mari kita lakukan pembersihan data. Untuk missing values pada kolom `Book-Author` dan `Publisher`, kita akan menghapusnya. Sedangkan untuk data yang tidak valid pada kolom `Year-Of-Publication`, kita akan menghapus data yang bukan rentang dari tahun 1960 hingga 2025. Hal ini dilakukan karena data yang tidak valid tersebut hanya sedikit dan tidak akan mempengaruhi hasil rekomendasi.

# %%
print("Missing Values in Books:\n", books.isnull().sum(), "\n")
print("Duplicated Data in Books:\n", books.duplicated().sum())

# %%
# Handle invalid 'Year-Of-Publication' values
books['Year-Of-Publication'] = pd.to_numeric(
    books['Year-Of-Publication'], errors='coerce')
books = books.dropna(subset=['Year-Of-Publication'])
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
# Filter books published between 1960 and 2025
books = books[(books["Year-Of-Publication"] >= 1960) &
              (books["Year-Of-Publication"] <= 2025)]

books = books.dropna()
books = books.drop_duplicates()

# %% [markdown]
# Setelah selesai melakukan pemberishan, mari kita cek sekali lagi apakah masih terdapat missing values atau tidak

# %%
# Missing Values data books
books.isnull().sum()

# %%
# Duplicated data books
books.duplicated().sum()

# %%
# Mengecek nilai unik di kolom 'Year-Of-Publication'
print(books['Year-Of-Publication'].unique())

# %% [markdown]
# Sekarang, data sudah bersih mari kita lakukan visualisasi data untuk melihat distribusi data kembali.

# %% [markdown]
# #### Advanced EDA After Cleaning

# %%
plt.figure(figsize=(10, 5))
sns.histplot(books['Year-Of-Publication'], bins=50, kde=True)
plt.xlabel("Year of Publication")
plt.ylabel("Count")
plt.title("Distribution of Book Publication Years")
plt.show()

# %% [markdown]
# Sekarang kita dapat melihat distribusi data yang lebih baik setelah melakukan cleaning data. Distribusi data pada kolom `Year-Of-Publication` menunjukan rentang tahun yang lebih konsisten, dengan konsentrasi data pada tahun 2000an.

# %%
# Top 10 Publisher
top_publishers = books['Publisher'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_publishers.values, y=top_publishers.index,
            palette="viridis", hue=top_publishers.index)
plt.xlabel("Number of Books")
plt.ylabel("Publisher")
plt.title("Top 10 Publishers by Number of Books")

for i, v in enumerate(top_publishers.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()

# %% [markdown]
# Setelah dilakukan cleaning data, publisher `Harlequin` masih menjadi publisher yang paling banyak menerbitkan buku dengan jumlah 7534 buku.

# %%
# Top 10 Author
top_authors = books['Book-Author'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_authors.values, y=top_authors.index,
            palette="viridis", hue=top_authors.index)
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.title("Top 10 Authors by Number of Books")

for i, v in enumerate(top_authors.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()


# %% [markdown]
# Begitupun dengan author `Agatha Christie` masih menjadi author yang paling banyak menerbitkan buku dengan jumlah 594 buku meskipun mengalami penurunan.

# %% [markdown]
# ### Ratings.csv
#
# Berdasarkan EDA yang telah dilakukan, tidak terdapat missing values dan duplicated data pada file ratings.csv. Namun, terdapat 716.109 data rating 0, artinya tidak relevan karena user tersebut belum memberikan rating pada buku tersebut. Data ini tidak relevan untuk digunakan dalam sistem rekomendasi, maka kita akan menghapus data tersebut.

# %%
print("Missing Values in Ratings:\n", ratings.isnull().sum(), "\n")
print("Duplicated Data in Ratings:\n", ratings.duplicated().sum())

# %%
# Drop rows where 'Book-Rating' is 0
ratings = ratings[ratings['Book-Rating'] != 0]
ratings.shape

# %%
ratings.describe()

# %% [markdown]
# Setelah melakukan pembersihan data, kita dapat melihat bahwa rata-rata rating yang diberikan oleh user adalah 7.6.

# %% [markdown]
# #### Advanced EDA after cleaning

# %%
plt.figure(figsize=(10, 8))
ax = sns.countplot(x=ratings["Book-Rating"],
                   palette="coolwarm", hue=ratings["Book-Rating"])
plt.xlabel("Book Rating")
plt.ylabel("Count")
plt.title("Distribution of Book Ratings")
plt.xticks(range(11))

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# %% [markdown]
# Distribusi rating yang diberikan oleh user menunjukkan bahwa mayoritas user memberikan rating 8, diikuti oleh rating 10 dan rating 7.

# %% [markdown]
# ### Users.csv
#
# Berdasarkan EDA yang telah dilakukan, terdapat missing values pada kolom `Age` sebanyak 110.762 data. Selain itu, data pada kolom `Age` memiliki rentang nilai yang tidak masuk akal yaitu dari 0 hingga 244. Kita akan melakukan cleaning data pada kolom `Age` dengan menghapus data yang tidak relevan dan mengisi missing values dengan median.

# %%
print("Missing Values in Users:\n", users.isnull().sum(), "\n")
print("Duplicated Data in Users:\n", users.duplicated().sum())

# %% [markdown]
# Kita hanya akan menggunakan data pengguna yang berusia antara 5 hingga 80 tahun, karena data di luar rentang tersebut tidak relevan dan mungkin merupakan data yang kurang valid.

# %%
# Remove age outliers
users = users[(users['Age'] >= 5) & (users['Age'] <= 80)]

# Impute missing age values with the median
median_age = round(users['Age'].median())
users['Age'] = users['Age'].fillna(median_age)


# %%
users.describe()

# %% [markdown]
# Berdasarkan data statistik deskriptif, kita dapat melihat bahwa usia pengguna memiliki rentang nilai yang lebih masuk akal yaitu dari 5 hingga 80 tahun. Selain itu, rata-rata usia pengguna adalah 34 tahun.

# %% [markdown]
# #### Advanced EDA After Cleaning

# %%
plt.figure(figsize=(10, 5))
sns.histplot(users["Age"].dropna(), bins=30, kde=True)
plt.xlabel("User Age")
plt.ylabel("Count")
plt.title("Distribution of User Ages")
plt.show()

# %% [markdown]
# Distribusi data usia pengguna menunjukkan bahwa mayoritas pengguna berusia antara 20-30 tahun. Setelah dilakukan cleaning data, top 10 lokasi pengguna yang paling banyak terdaftar dalam dataset adalah `london, england, united kingdom` dengan jumlah 2506 pengguna.

# %%
top_locations = users["Location"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_locations.values, y=top_locations.index,
            palette="magma", hue=top_locations.index)
plt.xlabel("Number of Users")
plt.ylabel("Location")
plt.title("Top 10 User Locations")

for i, v in enumerate(top_locations.values):
    plt.text(v + 1, i, str(v), color='black', va='center')

plt.show()

# %% [markdown]
# # Model Development dengan Content Based Filtering
#
# Pada tahap ini kita akan membuat model rekomendasi untuk content based filtering dengan TF-IDF berdasarkan kombinasi dari judul buku, penulis, dan penerbit. Kita akan menggunakan data dari file books.csv yang telah dibersihkan sebelumnya.

# %%
books.info()

# %% [markdown]
# Dikarenakan data yang digunakan terlalu besar, kita akan menggunakan 85.000 data acak dari file books.csv untuk membuat model rekomendasi.

# %%
NUM_ROWS = 85000
books_cbf = books.copy()
books_cbf = books_cbf.sample(NUM_ROWS, random_state=42).reset_index(drop=True)

# %%
books_cbf.sample(5)

# %%
# Combine 'Book-Title', 'Book-Author', and 'Publisher' into 'content'
books_cbf["content"] = books_cbf["Book-Title"] + " " + \
    books_cbf["Book-Author"] + " " + books_cbf["Publisher"]
books_cbf[["Book-Title", "content"]].head()


# %% [markdown]
# ## TF-IDF
#
# Kita akan menggunakan TF-IDF untuk mengubah kombinasi fitur penulis, penerbit dan judul buku menjadi representasi numeric, sehingga nantinya akan dihitung menggunakan cosine similarity untuk mendapatkan rekomendasi.

# %%
# Vectorize
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_cbf['content'].astype(str))

# %% [markdown]
# ## Cosine Similarity
#
# Setelah fitur diubah menjadi representasi numerik, langkah selanjutnya adalah menghitung kemiripan antar buku menggunakan Cosine Similarity. Semakin mirip dua buku berdasarkan fitur mereka, semakin tinggi skor kemiripan. Kita menggunakan parameter dense_output=False untuk menghemat memori, agar notebook tidak crash.

# %%
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)

# %% [markdown]
# ## Get Recommendation
#
# Buat fungsi untuk memberikan rekomendasi buku berdasarkan judul yang diberikan. Fungsi ini akan mencari buku yang memiliki skor kemiripan tertinggi dengan buku yang dicari.

# %%


def get_recommendations(book_title, books, cosine_sim, k=10):
    idx = books[books["Book-Title"].str.lower().str.strip() ==
                book_title.lower().strip()].index
    if len(idx) == 0:
        return "Buku tidak ditemukan dalam dataset!"

    idx = idx[0]
    similarity_scores = cosine_sim[idx].toarray().flatten()
    similar_indices = np.argsort(similarity_scores)[::-1][1:k+1]

    recommended_books = books.iloc[similar_indices].copy()
    recommended_books["Similarity"] = similarity_scores[similar_indices]

    return recommended_books

# %% [markdown]
# Kita akan mencoba memberikan rekomendasi buku berdasarkan judul `High Society` dengan 10 rekomendasi buku yang memiliki skor kemiripan tertinggi.


# %%
BOOK_TITLE_PREDICT = 'High Society'
recommended_books = get_recommendations(
    BOOK_TITLE_PREDICT, books_cbf, cosine_sim, k=10)
recommended_books

# %% [markdown]
# Yeay! Akhirnya kita berhasil membuat sistem rekomendasi content based filetering dengan TF-IDF. Kita telah berhasil memberikan rekomendasi buku berdasarkan judul dengan memanfaatkan fitur-fitur yang ada pada dataset books.csv.

# %% [markdown]
# ## Content Based Filtering Evaluation
#
# Untuk mengevaluasi model content based filtering yang telah dibuat, kita akan menggunakan metode evaluasi precision at k. Kita akan membandingkan rekomendasi yang diberikan oleh model dengan rekomendasi yang seharusnya diberikan.

# %%


def precision_at_k(recommended_books, base_book, k=5):
    relevant = 0
    top_k_recommendations = recommended_books.head(k)

    for _, book in top_k_recommendations.iterrows():
        if (book['Book-Title'] == base_book['Book-Title']) or (book['Book-Author'] == base_book['Book-Author']) or (book['Publisher'] == base_book['Publisher']):
            relevant += 1

    return relevant / k


# ** Evaluasi Precision@K **
base_book = books_cbf[books_cbf['Book-Title'].str.lower()
                      == BOOK_TITLE_PREDICT.lower().strip()].iloc[0]
precision = precision_at_k(recommended_books, base_book, k=10)
print(f"Precision@10: {precision:.2f}")

print("\nEvaluation Detail:")
print("Book to Search:", BOOK_TITLE_PREDICT)
print("\nTop 10 books with title, author and publisher similarities:")
for _, data in recommended_books.head(10).iterrows():
    print(f"Title: {data['Book-Title']}")
    print(f"Author: {data['Book-Author']}")
    print(f"Publisher: {data['Publisher']}")
    print("-" * 50)

# %% [markdown]
# Hasil precision menunjukan angka 0.80 dari 10 rekomendasi yang diberikan oleh model, 8 di antaranya adalah rekomendasi yang relevan. Hal ini menunjukkan bahwa model content based filtering yang telah dibuat memiliki tingkat akurasi yang baik dalam memberikan rekomendasi buku.

# %% [markdown]
# # Model Development dengan Collaborative Filtering

# %% [markdown]
# ## Data Preaparation
#
# Sebelum melakukan pembuatan model, kita harus memastikan bahwa data telah bersih dari duplicated data dan missing values

# %%
# Missing Values data Rating
ratings.isnull().sum()

# %% [markdown]
# Tidak terdapat missing values pada file rating.csv

# %%
ratings.duplicated().sum()

# %% [markdown]
# ### Encoding User-ID dan ISBN
#
# Pada tahap ini kita akan melakukan encoding pada kolom User-ID dan ISBN untuk mempermudah proses modeling. Hal ini dilakukan dengan mengubah User-ID dan ISBN menjadi kategori yang unik.

# %%
# Encoding User-ID and ISBN
user_ids = ratings['User-ID'].unique().tolist()
book_ids = ratings['ISBN'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}

# %% [markdown]
# Encoding dilakukan untuk mengubah data kategorikal, seperti User-ID dan ISBN, menjadi format numerik agar dapat diproses oleh model machine learning

# %%
# Mapping to dataframe
ratings['user'] = ratings['User-ID'].map(user_to_user_encoded)
ratings['book'] = ratings['ISBN'].map(book_to_book_encoded)

num_users = len(user_to_user_encoded)
num_books = len(book_to_book_encoded)

print(f"total user encoded: {num_users}")
print(f"total book encoded: {num_books}")

# %% [markdown]
# Terdapat 77.805 unique User-ID dan 185.973 unique ISBN dalam dataset

# %% [markdown]
# ### Normalisasi Rating
# Mengubah nilai rating ke skala antara 0 dan 1 untuk memudahkan proses pelatihan model.

# %%
min_rating = ratings['Book-Rating'].min()
max_rating = ratings['Book-Rating'].max()
ratings['Book-Rating'] = (ratings['Book-Rating'] -
                          min_rating) / (max_rating - min_rating)

# %% [markdown]
# ## Membagi data menjadi training dan validation
#
# Membagi dataset menjadi:
# - 80% data training untuk melatih model.
# - 20% data validation untuk mengevaluasi performa model selama pelatihan.
#
# - x_train dan x_val: Berisi pasangan user dan book.
# - y_train dan y_val: Berisi rating yang menjadi target.

# %% [markdown]
# Acak data sebelum membagi dataset menjadi training dan validation untuk menghindari bias.

# %%
ratings = ratings.sample(frac=1, random_state=42)
ratings

# %%
train_indices = int(0.8 * ratings.shape[0])
x_train, x_val = ratings[['user', 'book']].values[:train_indices], ratings[[
    'user', 'book']].values[train_indices:]
y_train, y_val = ratings['Book-Rating'].values[:
                                               train_indices], ratings['Book-Rating'].values[train_indices:]

# %% [markdown]
# ## Membangun Model Collaborative Filtering menggunakan TensorFlow
# Membuat arsitektur model dengan:
#
# - Embedding Layer:
# Digunakan untuk memetakan user dan book ke dalam vektor berdimensi rendah.
# - Dot Product:
# Menghitung kesamaan antara vektor user dan book.
# - Bias:
# Menambahkan nilai bias ke hasil prediksi untuk menangkap kecenderungan individu.

# %%


class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, embeddings_initializer='he_normal')
        self.book_embedding = layers.Embedding(
            num_books, embedding_size, embeddings_initializer='he_normal')
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        user_bias = self.user_bias(inputs[:, 0])
        book_bias = self.book_bias(inputs[:, 1])

        dot_product = tf.tensordot(user_vector, book_vector, 2)
        output = dot_product + user_bias + book_bias
        return tf.nn.sigmoid(output)

# %% [markdown]
# ## Menyiapkan model
#
# Model ini menggunakan Mean Squared Error (MSE) sebagai loss function, Adam sebagai optimizer, dan Root Mean Squared Error (RMSE) sebagai metrics evaluation.


# %%
embedding_size = 35
model = RecommenderNet(num_users, num_books, embedding_size)

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# %% [markdown]
# - Ukuran Embedding:
# Mengatur dimensi embedding untuk user dan book.
# - Kompilasi Model:
#   - Loss: Menggunakan mean squared error sebagai loss function karena rating bersifat continuous.
#   - Optimizer: Adam digunakan untuk memperbarui bobot secara efisien.
#

# %% [markdown]
# ## Melatih Model
# Langkah berikutnya, mulailah proses training.

# %%
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_data=(x_val, y_val)
)

# %% [markdown]
# ## Visualisasi Hasil Training

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Training vs Validation RMSE')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# %% [markdown]
# Berdasarkan visualisasi data di atas, kita dapat melihat bahwa model telah belajar dengan baik dan loss-nya menurun seiring dengan epoch yang berjalan. Model memiliki root mean squared error sebesar 0.1955 pada data training dan 0.2168 pada data validation. Hal ini menunjukkan bahwa model memiliki performa yang baik dalam memprediksi rating buku.

# %% [markdown]
# ## Membuat Rekomendasi untuk Pengguna
#
# Membuat function untuk mendapatkan rekomendasi bagi pengguna
# 1. Mengambil buku yang belum dibaca oleh pengguna.
# 2. Menggunakan model untuk memprediksi rating setiap buku yang belum dibaca.
# 3. Memilih buku dengan rating prediksi tertinggi sebagai rekomendasi.

# %%


def get_book_recommendations(user_id, model, ratings, books, user_to_user_encoded, book_to_book_encoded, num_recommendations=10):
    user_idx = user_to_user_encoded[user_id]
    user_book_ids = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
    all_book_ids = set(book_to_book_encoded.keys())
    unvisited_book_ids = list(all_book_ids - user_book_ids)
    unvisited_books_idx = [book_to_book_encoded[bid]
                           for bid in unvisited_book_ids]

    user_book_pairs = np.hstack((
        np.full((len(unvisited_books_idx), 1), user_idx),
        np.array(unvisited_books_idx).reshape(-1, 1)
    ))

    predicted_ratings = model.predict(user_book_pairs).flatten()
    # Ambil lebih banyak untuk menghindari error
    top_indices = predicted_ratings.argsort()[-num_recommendations * 2:][::-1]
    recommended_book_ids = [unvisited_book_ids[idx] for idx in top_indices]
    predicted_ratings = predicted_ratings[top_indices]

    recommendations = []
    for book_id, pred_rating in zip(recommended_book_ids, predicted_ratings):
        book_info = books[books['ISBN'] == book_id]
        if book_info.empty:
            continue
        book_info = book_info.iloc[0]

        recommendations.append({
            'ISBN': book_id,
            'Title': book_info['Book-Title'],
            'Author': book_info['Book-Author'],
            'Year': book_info['Year-Of-Publication'],
            'Publisher': book_info['Publisher'],
            'Predicted Rating': float(pred_rating * 10)
        })

        if len(recommendations) >= num_recommendations:
            break

    return recommendations

# %% [markdown]
# ## Test Rekomendasi
#
# Menampilkan 10 buku rekomendasi untuk pengguna tertentu berdasarkan prediksi rating tertinggi.


# %%
# Get recommendations for a user
user_id = 276760
recommendations = get_book_recommendations(
    user_id=user_id,
    model=model,
    ratings=ratings,
    books=books,
    user_to_user_encoded=user_to_user_encoded,
    book_to_book_encoded=book_to_book_encoded
)

# Show recommendations
print(f"\nUser profiles: ")
print("-" * 80)
user_profile = users[users['User-ID'] == user_id].iloc[0]
print(f"User ID: {user_id}")
print(f"Location: {user_profile['Location']}")
print(f"Age: {user_profile['Age']}")
print("-" * 80)
print(f"\nTop 10 Book Recommendations for User {user_id}:")
print("-" * 80)
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['Title']} by {rec['Author']}")
    print(f"   Year: {rec['Year']}, Publisher: {rec['Publisher']}")
    print(f"   Predicted Rating: {rec['Predicted Rating']:.2f}")
    print("-" * 80)


# %% [markdown]
# Kita sudah berhasil membuat model collaborative filtering menggunakan TensorFlow dan memberikan rekomendasi buku berdasarkan prediksi rating tertinggi. Model ini dapat memberikan rekomendasi buku kepada pengguna dengan akurasi yang sangat baik.
