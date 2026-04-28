import googleapiclient.discovery
import pandas as pd
import time
import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE # Pastikan sudah: pip install imbalanced-learn

# Download resource NLTK
nltk.download('punkt')

# ==========================================================
# 1. TAHAP SCRAPING (YOUTUBE)
# ==========================================================
print("--- Memulai Scraping ---")
API_KEY = 'AIzaSyCyJol93iiFLz9aPs0Mj26cUJ029n8fCFI' 
videoId = "L4TWcOnUF9E"

youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)
all_comments_data = []
nextPageToken = None

for _ in range(2): 
    request = youtube.commentThreads().list(
        part="snippet", videoId=videoId, maxResults=100, pageToken=nextPageToken
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        all_comments_data.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt']])
    nextPageToken = response.get('nextPageToken')
    if not nextPageToken: break

df_raw = pd.DataFrame(all_comments_data, columns=['text', 'author', 'publishedAt'])
df_raw.to_csv('Youtube_scrap.csv', index=False, sep=';', encoding='utf-8')
print(f"Berhasil scrap {len(df_raw)} data.")

# ==========================================================
# 2. TAHAP PREPROCESSING & LABELING
# ==========================================================
print("\n--- Memulai Preprocessing & Labeling ---")
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r'http\S+|www\.\S+|[@#]\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df_clean = pd.read_csv('Youtube_scrap.csv', sep=';')
df_clean['message'] = df_clean['text'].apply(clean_text)
df_clean = df_clean[df_clean['message'] != ''] 

pos_words = ['baik', 'bagus', 'mantap', 'aman', 'untung', 'cuan', 'mudah', 'resmi', 'legal', 'puas']
neg_words = ['buruk', 'jelek', 'rugi', 'nipu', 'penipuan', 'scam', 'error', 'bodong', 'kecewa', 'lambat']

def get_label(text):
    score = sum(1 for w in pos_words if w in text) - sum(1 for w in neg_words if w in text)
    return 'positif' if score > 0 else ('negatif' if score < 0 else 'netral')

df_clean['label'] = df_clean['message'].apply(get_label)
df_clean.to_csv('AfterPrepro_Labeled.csv', index=False, sep=';', encoding='utf-8')

# ==========================================================
# 3. TAHAP TF-IDF & SMOTE (PENYEIMBANGAN DATA)
# ==========================================================
print("\n--- Memulai TF-IDF & SMOTE ---")
df_final = pd.read_csv('AfterPrepro_Labeled.csv', sep=';')
df_final['message'] = df_final['message'].fillna('').astype(str)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df_final['message']).toarray()

# Encode label ke angka
y_encoded, label_map = pd.factorize(df_final['label'])

# Jalankan SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, y_encoded)

print(f"Data Awal: {np.bincount(y_encoded)}")
print(f"Data Setelah SMOTE: {np.bincount(y_res)}")

# ==========================================================
# 4. MODEL LMKNN
# ==========================================================
class LMKNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classes = np.unique(y)
    def predict(self, X_test):
        preds = []
        for x_q in X_test:
            class_dists = []
            for c in self.classes:
                X_c = self.X_train[self.y_train == c]
                if len(X_c) == 0:
                    class_dists.append(1e9)
                    continue
                dists = np.linalg.norm(X_c - x_q, axis=1)
                k_indices = np.argsort(dists)[:min(self.k, len(X_c))]
                local_mean = np.mean(X_c[k_indices], axis=0)
                class_dists.append(np.linalg.norm(x_q - local_mean))
            preds.append(self.classes[np.argmin(class_dists)])
        return np.array(preds)

# Split data yang sudah seimbang
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = LMKNN(k=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================================
# 5. EVALUASI & VISUALISASI
# ==========================================================
print(f"\nAkurasi Final (SMOTE + LMKNN): {accuracy_score(y_test, y_pred):.2%}")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=label_map))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=label_map, yticklabels=label_map)
plt.title('Confusion Matrix: LMKNN + SMOTE')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()