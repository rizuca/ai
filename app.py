import streamlit as st
import pandas as pd
import re
import joblib
import json

# Set page config
st.set_page_config(
    page_title="GuardAI - Moderasi Konten",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS: Premium SaaS Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1e293b;
}

/* Background */
.stApp {
    background-color: #f8fafc;
}

/* ── Hide unwanted Streamlit UI ── */
#MainMenu, footer, .stDeployButton {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Header container padding */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 800px !important;
}

/* Custom Hero Section */
.hero-section {
    text-align: center;
    padding: 1rem 0 3rem 0;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 1rem;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Text Area ── */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 15px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) inset !important;
    transition: all 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}
.stTextArea label {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 1rem !important;
}

/* ── Button ── */
.stButton button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1) !important;
    margin-top: 0.5rem;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.2) !important;
    background: #1e293b !important;
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    text-align: center;
}
div[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: #64748b !important;
    font-size: 0.9rem !important;
    margin-bottom: 8px;
    justify-content: center;
}
div[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #0f172a !important;
    font-size: 1.4rem !important;
}
div[data-testid="stMetricDelta"] {
    justify-content: center;
}

/* ── Result Card Wrapper ── */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    height: 100%;
}
.result-card h4 {
    margin-top: 0;
    margin-bottom: 20px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #334155;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 12px;
}

/* ── Tag Badges ── */
.tag-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 10px;
    margin-bottom: 10px;
}
.tag-label {
    font-weight: 600;
    color: #475569;
    font-size: 0.95rem;
}
.badge-safe {
    background: #dcfce7;
    color: #166534;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}
.badge-danger {
    background: #fee2e2;
    color: #991b1b;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# MODEL & DATA PIPELINE (Load Pre-trained)
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat otak AI...")
def load_model():
    try:
        clf = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        with open('slang_dict.json', 'r', encoding='utf-8') as f:
            slang_dict = json.load(f)
    except FileNotFoundError:
        st.error("File model.pkl atau vectorizer.pkl tidak ditemukan! Silakan jalankan 'train.py' terlebih dahulu.")
        st.stop()
        
    targets = ['HS', 'Abusive', 'HS_Individual', 'HS_Group', 'HS_Religion', 'HS_Race', 'HS_Physical', 'HS_Gender', 'HS_Other', 'HS_Weak', 'HS_Moderate', 'HS_Strong']
    return clf, vectorizer, slang_dict, targets

# Initialize Model instantly
clf, vectorizer, slang_dict, target_names = load_model()

def clean_text(text, slang_dict):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    return ' '.join(words)


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ GuardAI")
    st.caption("Sistem Moderasi Konten Cerdas")
    st.divider()
    st.markdown("""
    **👥 Tim Pengembang (Kelompok 2)**
    - Maria Winarni Br Silitonga
    - Rifqi Putra Winanda
    - Mohd. Rafiif Albani
    """)
    st.divider()
    st.markdown("""
    **📚 Mata Kuliah**  
    Kecerdasan Buatan
    
    **👨‍🏫 Dosen Pengampu**  
    Kana Saputra S., S.Pd., M.Kom
    """)
    st.divider()
    st.markdown("""
    **🤖 Info Model AI**  
    - **Dataset:** 41.617 teks
    - **Metode:** Naïve Bayes
    - **Status:** Di-deploy (Pre-trained)
    """)
    st.caption("© 2026 Universitas Negeri Medan")


# ══════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Sistem Klasifikasi Ujaran Kebencian</div>
    <div class="hero-subtitle">
        Aplikasi moderasi konten cerdas berbasis <b>Multinomial Naïve Bayes</b>. 
        Deteksi <i>Hate Speech</i> dan bahasa kasar secara otomatis untuk menjaga komunitas digital yang positif.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Section ──
user_input = st.text_area(
    "Masukkan teks atau tweet dari media sosial:",
    height=140,
    placeholder="Contoh: Dasar kafir, otak lu di mana sih?..."
)

analyze_btn = st.button("Pindai Teks Sekarang", use_container_width=True)

# ── Results Section ──
if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong. Ketik sesuatu terlebih dahulu.")
    else:
        # Preprocess using the slang dict
        cleaned = clean_text(user_input, slang_dict)

        # Predict using Pre-trained Machine Learning Model
        X_input = vectorizer.transform([cleaned])
        prediction = clf.predict(X_input)[0]  # Array of 0s and 1s
        
        # Map prediction to dictionary
        res = dict(zip(target_names, prediction))

        is_hs = bool(res.get('HS', 0))
        is_abusive = bool(res.get('Abusive', 0))
        is_individual = bool(res.get('HS_Individual', 0))
        is_group = bool(res.get('HS_Group', 0))
        is_religion = bool(res.get('HS_Religion', 0))
        is_race = bool(res.get('HS_Race', 0))
        is_physical = bool(res.get('HS_Physical', 0))

        # Severity Logic
        if res.get('HS_Strong', 0) or (is_hs and (is_group or is_religion)):
            severity = "TINGGI"
        elif res.get('HS_Moderate', 0) or is_hs:
            severity = "SEDANG"
        elif is_abusive:
            severity = "RENDAH"
        else:
            severity = "AMAN"

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>📊 Hasil Analisis AI</h3>", unsafe_allow_html=True)

        # ── Top 3 Metric Cards ──
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Hate Speech",
                "TERDETEKSI" if is_hs else "AMAN",
                "- Ujaran Kebencian" if is_hs else "+ Bebas Kebencian",
                delta_color="inverse" if is_hs else "normal"
            )
        with m2:
            st.metric(
                "Abusive Language",
                "TERDETEKSI" if is_abusive else "AMAN",
                "- Kata Kasar" if is_abusive else "+ Bahasa Sopan",
                delta_color="inverse" if is_abusive else "normal"
            )
        with m3:
            st.metric(
                "Level Keparahan",
                severity,
                "- Butuh Tindakan" if severity not in ["AMAN", "RENDAH"] else "+ Terkendali",
                delta_color="inverse" if severity not in ["AMAN", "RENDAH"] else "normal"
            )

        st.write("")
        st.write("")

        # ── Detail Analysis in 2 clean cards ──
        def tag_row(label, detected):
            badge = '<span class="badge-danger">Terdeteksi</span>' if detected else '<span class="badge-safe">Aman</span>'
            return f'<div class="tag-item"><span class="tag-label">{label}</span>{badge}</div>'

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"""
            <div class="result-card">
                <h4>🎯 Target Serangan</h4>
                {tag_row("Individu", is_individual)}
                {tag_row("Kelompok", is_group)}
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="result-card">
                <h4>⚖️ Kategori SARA</h4>
                {tag_row("Agama (Religion)", is_religion)}
                {tag_row("Ras / Etnis (Race)", is_race)}
                {tag_row("Fisik (Physical)", is_physical)}
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # ── Preprocessing Details (collapsible) ──
        with st.expander("🔧 Tampilkan Data Preprocessing (KDD)"):
            st.markdown(f"**Teks Mentah:** `{user_input}`")
            st.markdown(f"**Teks Bersih (Case Folding, Cleansing, Slang Normalization):** `{cleaned}`")
