# 🦴 Klasifikasi Tumor Tulang dari Citra X-Ray

Repositori ini berisi implementasi sistem **klasifikasi citra medis** untuk mendeteksi tumor tulang menggunakan citra X-Ray.
Proyek ini membandingkan tiga pendekatan:

* **Pendekatan klasik** (SVM, KNN, RF, ekstraksi fitur)
* **Pendekatan deep learning** (CNN)
* **Pendekatan Transfer Learning** (MobileNetV2, LoRA, EfficientNet-B0)

Model bertugas mengklasifikasikan citra ke dalam dua kelas:

* **Normal**
* **Tumor**

Tujuan proyek ini adalah membantu deteksi dini tumor tulang dan mendukung proses diagnosis medis secara otomatis, cepat, dan konsisten.

---
## 📂 Dataset

Dataset yang digunakan berasal dari Figshare:

**A Radiograph Dataset for the Classification, Localization and Segmentation of Primary Bone Tumors**
🔗 [https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398?file=50653575](https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398?file=50653575)

### Ringkasan Dataset

* Total citra: **3.746**
* Kelas: **Normal (1.873)**, **Tumor (1.873)**
* Jenis citra: Radiografi X-Ray
* Anotasi tambahan: bounding box & segmentation
* Format file: PNG / JPG

Dataset ini dirancang untuk mendukung penelitian klasifikasi, deteksi, dan segmentasi tumor tulang.


### Contoh Citra

<table> <tr> <td><img src="https://github.com/user-attachments/assets/1701c510-6bb8-4ade-b733-aeb63d5876a5" width="200"/></td> <td><img src="https://github.com/user-attachments/assets/846a60c0-20bc-4101-a603-2e6327bf6a26" width="200"/></td> <td><img src="https://github.com/user-attachments/assets/9d6a50d5-6588-444e-9e74-2e07bc810ac0" width="200"/></td> <td><img src="https://github.com/user-attachments/assets/1e5f1604-f2dd-4454-94a3-69fce358113a" width="200"/></td> </tr> <tr> <td align="center"><b>Sample 1 – Normal</b></td> <td align="center"><b>Sample 2 – Tumor</b></td> <td align="center"><b>Sample 3 – Normal</b></td> <td align="center"><b>Sample 4 – Tumor</b></td> </tr> </table>
---

## 📁 Struktur Proyek
```
project/
│
├── BTXRD/
│   ├── Normal/
│   ├── Tumor/
│
├── Project_Machine_Learning.ipynb
│
└── README.md
```

## 🔍 Alur Metode

### 1️⃣ Preprocessing

* Pengecekan citra rusak
* Resize → 224×224 piksel
* Normalisasi piksel (0–1)
* Data augmentation:

  * Flip
  * Rotasi
  * Zoom
* Filtering:

  * Median
  * Gaussian
  * Bilateral
  * Denoising

---

## 🧠 Ekstraksi Fitur (Pendekatan Klasik)

Tujuh metode ekstraksi fitur digunakan untuk mendapatkan representasi numerik dari citra:

| Metode          | Deskripsi                  |
| --------------- | -------------------------- |
| HOG             | Kontur & gradien           |
| LBP             | Tekstur lokal              |
| GLCM            | Tekstur statistik          |
| Histogram Warna | Distribusi intensitas RGB  |
| Hu Moments      | Bentuk global              |
| Edge Histogram  | Struktur tepi              |
| Gabor Filter    | Pola frekuensi & orientasi |

Semua fitur kemudian dinormalisasi sebelum masuk ke model ML.

---

## ⚙️ Model yang Digunakan

### Model Klasik

* **SVM (RBF)**
* **Random Forest**
* **KNN (k=5)**

### Model Deep Learning

* **CNN dari awal (baseline)**

### Model Transfer Learning

* **MobileNetV2 + LoRA Fine-Tuning** ← *Model terbaik*
* **EfficientNet-B0**
* **MobileViT_xxs**

---

## ▶️ Cara Menjalankan

### 1. Buka Notebook Utama

Seluruh proses—mulai dari preprocessing, ekstraksi fitur, pemodelan klasik, deep learning, hingga evaluasi—dapat dijalankan melalui file:

```
Project_Machine_Learning.ipynb
```

Untuk membuka notebook, jalankan:

```
jupyter notebook Project_Machine_Learning.ipynb
```


### 2. Pastikan Struktur Dataset Sesuai

Notebook akan membaca dataset dari folder berikut:

```
BTXRD/
    ├── Normal/
    ├── Tumor/
```

Pastikan folder dan nama subfolder sudah sesuai.

### 3. Jalankan Seluruh Cell Notebook

Setelah membuka notebook:

* Klik **Run All**
  atau
* Jalankan setiap sel dari atas ke bawah

Semua tahap pemrosesan dan pelatihan dilakukan langsung dari notebook tanpa memerlukan file `.py` terpisah.

---


## Hasil Eksperimen

### 🔵 Pendekatan Klasik — Performa Terbaik

**Random Forest + Histogram Warna → 74.10%**

Performa lainnya:

* SVM + LBP → **72.50%**
* SVM + HOG → **72.36%**
* KNN + LBP → **67.29%**

### 🔴 Pendekatan Deep Learning 

**CNN → 66.89%**

### 🔴 Pendekatan Transfer Learning 

**MobileNetV2 + LoRA → 73.70%**
**EfficientNet-B0 → 76.67%**
**MobileViT_xxs → 73.47%**


---
##  Tabel Perbandingan Akurasi dan Karakteristik Pendekatan


| Pendekatan         | Model                 | Akurasi  | Pendefinisian                                                                 |
|--------------------|------------------------|----------|--------------------------------------------------------------------------------|
| **Klasik**         | SVM + LBP              | 72.50%   | Menggunakan tekstur lokal (LBP) untuk klasifikasi sederhana dan stabil.       |
| **Klasik**         | KNN + HOG              | 67.42%   | Mengandalkan bentuk dan edge, namun sensitif terhadap noise dan kontras.      |
| **Klasik**         | Random Forest + RGB    | 74.10%   | Berdasarkan distribusi intensitas RGB dan menjadi terbaik di metode klasik.   |
| **Deep Learning**  | CNN (training from scratch) | 66.89% | Belajar fitur otomatis tetapi kurang efektif pada dataset kecil (overfitting).|
| **Transfer Learning** | MobileNetV2 + LoRA  | 73.70%   | Model efisien; LoRA meningkatkan adaptasi dengan parameter rendah.            |
| **Transfer Learning** | EfficientNet-B0     | 76.67%   | Memberikan akurasi tertinggi dan generalisasi terbaik di seluruh pendekatan.  |
| **Transfer Learning** | MobileViT_xxs       | 73.47%   | Hybrid CNN-Transformer, menangkap representasi global lebih baik.             |

Hasil evaluasi menunjukkan bahwa setiap pendekatan memiliki keunggulan berbeda, namun transfer learning memberikan performa paling tinggi dan stabil.

---

## 🧪 Analisis Kesalahan

Beberapa penyebab mis-klasifikasi:

| Masalah                 | Penyebab                      | Solusi                                          |
| ----------------------- | ----------------------------- | ------------------------------------------------|
| Normal → Tumor          | Bayangan atau citra gelap     | Normalisasi (CLAHE), augmentasi cahaya          |
| Tumor → Normal          | Lesi kecil sulit terdeteksi   | Multi-scale feature + Gabor / GLCM              |
| Semua Kelas             | Variasi kontras sangat tinggi | Histogram equalization agar piksel lebih seragam|
| Deep - Fine-tuning      | Overfitting fitur ImageNet    | Regularisasi lebih kuat, early stopping         |
| Deep - CNN baseline     | Downsampling menghapus detail | Multi-scale CNN, skip connection (U-Net)        |
| Transfer - Fine-tuning  | FT penuh terlalu agresif      | Freeze layer, differential learning rate        |
| Transfer - LoRA         | Lesi sangat kecil             | Multi-scale attention, Grad-CAM guided cropping |
| Noise -> salah kelas    | Adsanya struktur artefak      | Denoising(bilateral/median), masking area       |

---

## 📚 Sitasi

```
@dataset{primary_bone_tumor_xray,
  title={A Radiograph Dataset for the Classification, Localization and Segmentation of Primary Bone Tumors},
  publisher={Figshare},
  year={2024}
}
```


