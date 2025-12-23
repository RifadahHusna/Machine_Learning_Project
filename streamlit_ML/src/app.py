import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_model.h5"
REPORT_PATH = "classification_report.txt"
LABELS_PATH = "val_labels_preds.json"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Tumor"]

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Medical AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# THEME SWITCHER
# ==============================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🧠 Medical AI")
menu = st.sidebar.radio("Menu", ["Dashboard", "Prediction", "Report"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Theme Settings")
theme = st.sidebar.radio(
    "Pilih Mode Tampilan",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1
)
st.session_state.theme = theme
st.sidebar.caption("CNN Medical Image Classification")

# ==============================
# THEME CSS
# ==============================
if st.session_state.theme == "Dark":
    bg = "#020617"
    card = "#0f172a"
    text = "#e5e7eb"
    border = "#1e293b"
else:
    bg = "#f7f9fc"
    card = "#ffffff"
    text = "#0f172a"
    border = "#e5e7eb"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
    font-family: 'Segoe UI', sans-serif;
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #334155, #1e293b);
    color: {text};
}}
.title {{
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 20px;
}}
.subtitle {{
    font-size: 18px;
    margin-bottom: 10px;
}}
.card, .metric-card {{
    background: {card};
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
    border: 1px solid {border};
}}
.badge-normal {{
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 600;
}}
.badge-tumor {{
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: 600;
}}
footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model_func():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_func() if os.path.exists(MODEL_PATH) else None

# ==============================
# DASHBOARD
# ==============================
if menu == "Dashboard":
    st.markdown("<div class='title'>📊 Medical AI Dashboard</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    support = {}
    total_samples = 0

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] in CLASS_NAMES:
                    support[parts[0]] = int(parts[-1])
        total_samples = sum(support.values())

    with c1:
        st.markdown(f"<div class='metric-card'><b>Model Status</b><br>{'READY ✅' if model else 'NOT READY ❌'}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><b>Classes</b><br>{len(CLASS_NAMES)}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><b>Image Size</b><br>{IMG_SIZE[0]}×{IMG_SIZE[1]}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><b>Total Samples</b><br>{total_samples}</div>", unsafe_allow_html=True)

    st.markdown("### 📊 Dataset Overview")

    st.markdown("""
    <div class='card'>
    <b>Penjelasan Proyek:</b><br><br>
    Visualisasi ini merepresentasikan gambaran umum dataset citra medis yang digunakan
    dalam proyek klasifikasi tumor otak berbasis <i>Convolutional Neural Network (CNN)</i>.
    Dataset terdiri dari dua kelas utama, yaitu <b>Normal</b> dan <b>Tumor</b>, yang digunakan
    pada tahap pelatihan, validasi, dan evaluasi model.

    Analisis distribusi dan proporsi kelas dilakukan untuk memastikan keseimbangan data
    serta mendukung interpretasi performa model secara objektif. Informasi ini penting
    untuk memahami karakteristik data, potensi bias kelas, dan pengaruhnya terhadap
    akurasi serta keandalan sistem deteksi tumor yang dikembangkan.

    """, unsafe_allow_html=True)

    if support:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(3.8, 2.6))
            ax.bar(support.keys(), support.values())
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie(
                [support.get(c, 0) for c in CLASS_NAMES],
                labels=CLASS_NAMES,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)

# ==============================
# PREDICTION
# ==============================
elif menu == "Prediction":
    st.markdown("<div class='title'>🖼️ Image Prediction</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Single Testing", "📂 Batch Testing"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        file = st.file_uploader("Upload satu citra medis", type=["jpg", "png", "jpeg"])
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, width=350)
            img = image.resize(IMG_SIZE)
            x = np.expand_dims(np.array(img)/255.0, axis=0)
            pred = model.predict(x)[0][0]
            label = CLASS_NAMES[int(pred > 0.5)]
            conf = pred if pred > 0.5 else 1 - pred
            st.markdown(f"<span class='badge-{'normal' if label=='Normal' else 'tumor'}'>{label.upper()}</span>", unsafe_allow_html=True)
            st.progress(float(conf))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        files = st.file_uploader("Upload banyak citra", accept_multiple_files=True)
        if files:
            cols = st.columns(3)
            for i, f in enumerate(files):
                image = Image.open(f).convert("RGB")
                img = image.resize(IMG_SIZE)
                x = np.expand_dims(np.array(img)/255.0, axis=0)
                pred = model.predict(x)[0][0]
                label = CLASS_NAMES[int(pred > 0.5)]
                conf = pred if pred > 0.5 else 1 - pred
                with cols[i % 3]:
                    st.image(image, use_container_width=True)
                    st.markdown(f"<span class='badge-{'normal' if label=='Normal' else 'tumor'}'>{label.upper()}</span>", unsafe_allow_html=True)
                    st.progress(float(conf))
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# REPORT
# ==============================
elif menu == "Report":
    st.markdown("<div class='title'>📄 Classification Report</div>", unsafe_allow_html=True)
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            st.code(f.read())

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("© Medical AI Dashboard | Deep Learning")
