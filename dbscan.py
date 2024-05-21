import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from sklearn.cluster import DBSCAN
import mplcursors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_excel(r"data_text.xlsx")
data_cloud =pd.read_excel(r"data_wordcloud.xlsx")

st.title('Indentifikasi Pola dan Jenis Hotel dengan DBSCAN')
# Sidebar menu
menu = st.sidebar.selectbox('Pilih Menu:', ['EDA', 'DBSCAN Clustering'])
display_columns =  ['Attractions', 'Room', 'Dining', 'Renovation', 'CheckIn Instructions',
             'Special Instructions', 'HotelFacilities', 'Address',' cityName', ' HotelName']
def create_wordcloud_from_text(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud


# Mengatur tata letak kolom
columns = st.columns([2, 1])
left_column = columns[0]
right_column = columns[1]
# Fungsi untuk mengubah nilai 'HotelRating' menjadi skala numerik (1-5)
def rating(rating):
    if rating == 'OneStar':
        return 1
    elif rating == 'TwoStar':
        return 2
    elif rating == 'ThreeStar':
        return 3
    elif rating == 'FourStar':
        return 4
    elif rating == 'FiveStar':
        return 5
    else:
        return None  # Jika nilai tidak sesuai, kembalikan None
data_cloud[' HotelRating'] = data_cloud[' HotelRating'].apply(rating)
data_cloud = data_cloud.dropna()
def plot_rating_barplot(data):
    # Hitung frekuensi kemunculan setiap nilai rating
    rating_counts = data[' HotelRating'].value_counts().sort_index()

    # Buat plot bar menggunakan matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.index.astype(str), rating_counts.values, color='skyblue')
    plt.xlabel('Hotel Rating')
    plt.ylabel('Frequency')
    plt.title('Frequency of Hotel Ratings')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    return plt

# Menampilkan wordcloud di kolom kiri
if menu == 'EDA':
    input_worldCloud = st.selectbox('Pilih kolom teks:', display_columns)
    all_text = ' '.join(data_cloud[input_worldCloud])
    wordcloud = create_wordcloud_from_text(all_text)
    st.image(wordcloud.to_array(), use_column_width=True)
    st.pyplot(plot_rating_barplot(data_cloud))
# Meminta input dari pengguna
elif menu == 'DBSCAN Clustering':
    data[' HotelRating'] = data[' HotelRating'].apply(rating)
    data = data.dropna()
    input_eps = st.number_input("Input Nilai eps : ", min_value=0.1, step=0.1)
    input_minsample = st.number_input("Input Nilai Min Sample",min_value=1, step=1,format="%d")
# Add "OK" button
    if st.button("Run"):
    # Check if both eps and min_samples are provided
        if input_eps and input_minsample:

            # Memilih fitur numerik untuk di-scaling
            X_numerical = data[[' cityCode',' HotelRating','latitude', 'longitude', ' PinCode']]

            # Inisialisasi StandardScaler
            scaler = StandardScaler()

            X_numerical_scaled = scaler.fit_transform(X_numerical)

            # Inisialisasi TF-IDF Vectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Terapkan TF-IDF Vectorizer pada kolom teks
            X_text = tfidf_vectorizer.fit_transform(data['text'])
            # Gabungkan fitur numerik dan fitur teks
            X_combined = np.hstack((X_numerical_scaled, X_text.toarray()))

            # Melakukan clustering menggunakan DBSCAN
            dbscan = DBSCAN(eps=input_eps, min_samples=input_minsample)
            clusters = dbscan.fit_predict(X_combined)

        # Menghitung jumlah cluster yang terbentuk
            num_clusters = len(np.unique(clusters[clusters != -1]))  # Mengabaikan noise (cluster -1)

        # Menambahkan hasil clustering ke dataframe
            data['cluster'] = clusters

        # Hitung Silhouette Score
            if len(set(clusters)) > 1 and -1 in clusters:
                silhouette_avg = silhouette_score(X_combined[clusters != -1], clusters[clusters != -1])


            # Visualisasi scatter plot dengan Streamlit
                plt.figure(figsize=(10, 8))
                for cluster_label in np.unique(clusters):
                    if cluster_label == -1:
                        plt.scatter(X_combined[clusters == cluster_label, 0], X_combined[clusters == cluster_label, 1],
                                color='white', alpha=0.7, label='Noise', s=100)
                    else:
                        plt.scatter(X_combined[clusters == cluster_label, 0], X_combined[clusters == cluster_label, 1],
                                label=f'Cluster {cluster_label}', alpha=0.7, s=100)

            # Tambahkan label untuk sumbu x dan y
                plt.xlabel('')
                plt.ylabel('')
                plt.legend()
            # Menampilkan plot dan hasil clustering
                st.write("Jumlah Cluster yang Terbentuk:", num_clusters)
                st.write("Silhouette Score:", silhouette_avg)
                st.title('DBSCAN Clustering')
                st.pyplot(plt)
        # Menampilkan daftar data per klaster
                nilai = st.number_input("Masukkan nilai klaster: (Contoh: 3) : ")
                daftar_perklaster = data['cluster'] == nilai
                daftar = data[daftar_perklaster]
                st.write(daftar)
            else:
                st.write('tidak ada klaster yang terbentuk')
