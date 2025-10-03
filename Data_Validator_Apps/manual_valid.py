import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

st.set_page_config(
    page_title="Sentimen Validator",
    page_icon="🧠",
    layout="wide"
)

st.title('🧠 Sentimen Validator')
st.info("Create by : Nadhif Fauzil Adhim")

DB_FILE = "validator_progress.db"

def init_db():
    """Inisialisasi database SQLite dan membuat tabel jika belum ada."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS validation_progress (
            original_index INTEGER PRIMARY KEY,
            status_validasi TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_progress_to_db(original_index, status):
    """Menyimpan atau memperbarui progress validasi ke database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO validation_progress (original_index, status_validasi)
        VALUES (?, ?)
    ''', (int(original_index), status)) 
    conn.commit()
    conn.close()

def load_progress_from_db():
    """Memuat semua progress validasi dari database."""
    conn = sqlite3.connect(DB_FILE)
    try:
        df_progress = pd.read_sql_query("SELECT * FROM validation_progress", conn)
        if not df_progress.empty:
            df_progress.set_index('original_index', inplace=True)
            return df_progress['status_validasi']
    except pd.io.sql.DatabaseError:
        return pd.Series(dtype=str) 
    finally:
        conn.close()
    return pd.Series(dtype=str)


@st.cache_data
def load_data(filepath):
    """Memuat dan membersihkan data dari file CSV."""
    df = pd.read_csv(filepath)
    df['model_conf'] = pd.to_numeric(df['model_conf'], errors='coerce')
    df['label_model'] = df['label_model'].astype(str).str.strip()
    df['label_lexicon'] = df['label_lexicon'].astype(str).str.strip()
    df.dropna(subset=['model_conf', 'label_model', 'label_lexicon'], inplace=True)
    df.reset_index(inplace=True)
    return df

def analyze_dynamic_threshold(df, incorrect_indices):
    if not incorrect_indices:
        return None, None
    valid_indices = [idx for idx in incorrect_indices if idx in df.index]
    if not valid_indices:
        return None, None
    incorrect_df = df.loc[valid_indices].copy()
    if incorrect_df.empty:
        return None, None
    stats = incorrect_df['model_conf'].describe()
    recommended_threshold = stats.get('75%')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(incorrect_df['model_conf'], bins=20, kde=True, color='coral', ax=ax)
    ax.set_title('Distribusi Kepercayaan pada Kesalahan Model')
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Frekuensi')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if recommended_threshold is not None:
        ax.axvline(recommended_threshold, color='darkred', linestyle='-.',
                   label=f"Threshold Baru: {recommended_threshold:.2f}")
    ax.legend()
    return recommended_threshold, fig

def save_results(df):
    """Menyimpan data yang sudah divalidasi dan yang dihapus ke file terpisah."""
    validated_df = df[df['status_validasi'].isin(['positif', 'negatif', 'netral'])].copy()
    deleted_df = df[df['status_validasi'] == 'dihapus'].copy()
    validated_df.to_csv('validated_results.csv', index=False)
    deleted_df.to_csv('deleted_data.csv', index=False)
    return len(validated_df), len(deleted_df)

init_db() 

DATA_FILE = '../datasets/ambiguous_3class_10000.csv'
if not os.path.exists(DATA_FILE):
    st.error(f"File '{DATA_FILE}' tidak ditemukan.")
    st.stop()

full_data = load_data(DATA_FILE)

if 'validation_df' not in st.session_state:
    st.session_state.validation_df = full_data[full_data['status'] == 'ambiguous'].copy()
    progress_from_db = load_progress_from_db()
    st.session_state.validation_df = st.session_state.validation_df.set_index('index').join(progress_from_db).reset_index()

validated_so_far_df = st.session_state.validation_df.dropna(subset=['status_validasi'])
indobert_correct = 0
indobert_incorrect = 0
lexicon_correct = 0
lexicon_incorrect = 0
incorrect_model_indices = set(full_data[full_data['label_model'] != full_data['label_lexicon']]['index'])

for _, row in validated_so_far_df.iterrows():
    decision = row['status_validasi']
    original_index = row['index']
    if decision in ['positif', 'negatif', 'netral']:
        if decision == row['label_model']:
            indobert_correct += 1
            incorrect_model_indices.discard(original_index)
        else:
            indobert_incorrect += 1
            incorrect_model_indices.add(original_index)
        if decision == row['label_lexicon']:
            lexicon_correct += 1
        else:
            lexicon_incorrect += 1
    elif decision == 'dihapus':
        incorrect_model_indices.discard(original_index)


unprocessed_df = st.session_state.validation_df[st.session_state.validation_df['status_validasi'].isna()]
if not unprocessed_df.empty:
    st.session_state.current_index = unprocessed_df.index[0]
else:
    st.session_state.current_index = len(st.session_state.validation_df)

if st.session_state.validation_df.empty:
    st.warning("Tidak ada data 'ambiguous' untuk divalidasi.")
    st.stop()

total_items = len(st.session_state.validation_df)

with st.sidebar:
    st.header("🧠 Analisis Threshold")
    threshold, fig = analyze_dynamic_threshold(full_data, incorrect_model_indices)
    if threshold is not None and fig is not None:
        st.metric("Rekomendasi Threshold", value=f"{threshold:.4f}")
        st.pyplot(fig)
        st.info(f"Berdasarkan **{len(incorrect_model_indices)}** kasus kesalahan model.")
    else:
        st.warning("Tidak ada data kesalahan teridentifikasi.")

    st.header("📊 Progress & Aksi")
    processed_count = len(validated_so_far_df)
    st.progress(processed_count / total_items if total_items > 0 else 0)
    st.write(f"{processed_count} dari {total_items} data diproses.")

    if st.button("💾 Simpan Semua Hasil", use_container_width=True, type="primary"):
        v_count, d_count = save_results(st.session_state.validation_df)
        st.success(f"Disimpan! {v_count} data tervalidasi & {d_count} data dihapus.")

    st.header("📈 Statistik Validasi")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**IndoBERT**")
            st.metric(label="❌ Salah", value=indobert_incorrect)
            st.metric(label="✅ Benar", value=indobert_correct)
        with col2:
            st.markdown("**Lexicon**")
            st.metric(label="❌ Salah", value=lexicon_incorrect)
            st.metric(label="✅ Benar", value=lexicon_correct)


current_idx = st.session_state.current_index

if current_idx >= total_items:
    st.balloons()
    st.success("🎉 Selamat! Semua data telah diproses. Jangan lupa simpan hasil Anda di sidebar.")
    st.stop()

item = st.session_state.validation_df.loc[current_idx]
original_index = item['index']

st.header(f"Validasi Data ke-{processed_count + 1}/{total_items}")

with st.container(border=True):
    st.markdown("**Teks Tweet:**")
    st.info(f"_{item['full_text']}_")

col1, col2 = st.columns([0.6, 0.4])
with col1:
    with st.container(border=True):
        st.subheader("Analisis Sistem")
        c1, c2, c3 = st.columns(3)
        c1.metric("Label IndoBERT", str(item['label_model']).capitalize())
        c2.metric("Kepercayaan Model", f"{item['model_conf']:.2%}")
        c3.metric("Label Lexicon", str(item['label_lexicon']).capitalize())

with col2:
    with st.container(border=True):
        st.subheader("Putusan Anda")

        def process_validation(decision):
            save_progress_to_db(original_index, decision)
            st.session_state.validation_df.loc[current_idx, 'status_validasi'] = decision
            st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button('👍 Positif', use_container_width=True):
            process_validation('positif')
        if c2.button('👎 Negatif', use_container_width=True):
            process_validation('negatif')
        if c3.button('😐 Netral', use_container_width=True):
            process_validation('netral')

        c4, c5 = st.columns(2)
        if c4.button('⏭️ Lewati (Skip)', use_container_width=True):
            st.session_state.current_index += 1
            st.rerun()
        if c5.button('🗑️ Hapus Data', use_container_width=True):
            process_validation('dihapus')


