import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen Emas LMKNN", layout="wide", page_icon="💰")

# Custom CSS untuk tampilan profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; background-color: #e9ecef; border-radius: 5px 5px 0 0; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CLASS LMKNN (Local Mean K-Nearest Neighbor) ---
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
                k_idx = np.argsort(dists)[:min(self.k, len(X_c))]
                local_mean = np.mean(X_c[k_idx], axis=0)
                class_dists.append(np.linalg.norm(x_q - local_mean))
            preds.append(self.classes[np.argmin(class_dists)])
        return np.array(preds)

# --- FUNGSI HELPER ---
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r'http\S+|www\.\S+|[@#]\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def get_lexicon_label(text):
    pos_words = ['baik', 'bagus', 'mantap', 'aman', 'untung', 'cuan', 'mudah', 'resmi', 'legal', 'puas']
    neg_words = ['buruk', 'jelek', 'rugi', 'nipu', 'penipuan', 'scam', 'error', 'bodong', 'kecewa', 'lambat']
    score = sum(1 for w in pos_words if w in text) - sum(1 for w in neg_words if w in text)
    return 'positif' if score > 0 else ('negatif' if score < 0 else 'netral')

# --- HEADER ---
st.title("💰 Dashboard Analisis")
st.markdown("Implementasi Algoritma **LMKNN** dengan Penyeimbangan Data **SMOTE**")
st.divider()

# --- SIDEBAR (Scraping) ---
with st.sidebar:
    st.header("🛠️ Data Crawler")
    api_key = st.text_input("YouTube API Key", type="password", value="AIzaSyCyJol93iiFLz9aPs0Mj26cUJ029n8fCFI")
    video_id = st.text_input("Video ID", value="L4TWcOnUF9E")
    if st.button("Mulai Scraping Baru"):
        try:
            youtube = build('youtube', 'v3', developerKey=api_key)
            comments = []
            request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
            response = request.execute()
            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append([snippet['textDisplay'], snippet['authorDisplayName'], snippet['publishedAt']])
            
            df_new = pd.DataFrame(comments, columns=['text', 'author', 'publishedAt'])
            df_new['message'] = df_new['text'].apply(clean_text)
            df_new['label'] = df_new['message'].apply(get_lexicon_label)
            df_new.to_csv('AfterPrepro_Labeled.csv', index=False, sep=';')
            st.success("Scraping & Labeling Selesai!")
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN CONTENT ---
if os.path.exists('AfterPrepro_Labeled.csv'):
    df = pd.read_csv('AfterPrepro_Labeled.csv', sep=';')
    
    tab1, tab2, tab3 = st.tabs(["📈 Statistik Data", "🤖 Model LMKNN", "🔍 Uji Real-time"])

    with tab1:
        st.subheader("Distribusi Sentimen & Topik")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", len(df))
        col2.metric("Positif", len(df[df['label']=='positif']), "Cuan", delta_color="normal")
        col3.metric("Negatif", len(df[df['label']=='negatif']), "Rugi", delta_color="inverse")
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='label', palette='viridis', ax=ax)
            st.pyplot(fig)
        with c2:
            all_txt = ' '.join(df['message'].astype(str))
            wc = WordCloud(background_color='white', width=800, height=450).generate(all_txt)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc)
            ax_wc.axis('off')
            st.pyplot(fig_wc)

    with tab2:
        st.subheader("Pengaturan LMKNN + SMOTE")
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            k_neigh = st.slider("Nilai K (LMKNN)", 1, 10, 3)
            use_smote = st.checkbox("Aktifkan SMOTE", value=True)
            if st.button("🚀 Jalankan Training"):
                # Proses TF-IDF
                vec = TfidfVectorizer(max_features=1000)
                X = vec.fit_transform(df['message'].fillna('')).toarray()
                y, l_map = pd.factorize(df['label'])
                
                # Perbaikan SMOTE (Error k_neighbors)
                if use_smote:
                    min_samples = pd.Series(y).value_counts().min()
                    # k_neighbors SMOTE tidak boleh >= n_samples kelas terkecil
                    smote_k = min(5, min_samples - 1) if min_samples > 1 else 1
                    if min_samples > 1:
                        sm = SMOTE(random_state=42, k_neighbors=smote_k)
                        X, y = sm.fit_resample(X, y)
                        st.info(f"SMOTE Aktif (k={smote_k})")
                    else:
                        st.warning("Data terlalu sedikit untuk SMOTE!")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LMKNN(k=k_neigh)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state['acc'] = accuracy_score(y_test, y_pred)
                st.session_state['cm'] = confusion_matrix(y_test, y_pred)
                st.session_state['l_map'] = l_map

        with col_b:
            if 'acc' in st.session_state:
                st.metric("Akurasi Model", f"{st.session_state['acc']:.2%}")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(st.session_state['cm'], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=st.session_state['l_map'], yticklabels=st.session_state['l_map'])
                st.pyplot(fig_cm)

    with tab3:
        st.subheader("Uji Kalimat Baru")
        input_user = st.text_input("Ketik komentar di sini:")
        if input_user:
            clean_input = clean_text(input_user)
            res = get_lexicon_label(clean_input)
            if res == 'positif': st.success(f"Hasil: {res.upper()} 😊")
            elif res == 'negatif': st.error(f"Hasil: {res.upper()} 😡")
            else: st.warning(f"Hasil: {res.upper()} 😐")

else:
    st.warning("Data belum tersedia. Silakan lakukan 'Scraping' di Sidebar terlebih dahulu.")