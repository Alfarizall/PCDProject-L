# ♻️ PCDProject-L

Klasifikasi Sampah Otomatis 🗑️ menggunakan Ekstraksi Fitur Citra dan Machine Learning (KNN & SVM)

---



### 👨‍🔧 Kelompok C6
```sh
- Muhammad Alfarizal Hafidz   - 152023050
- Muhammad Naufal Shidqi A    - 152023053
- Gamma Dzauqy Al-Banna       - 152023112
```
## 📁 Struktur Folder

```bash
PCDProject-L/
├── dataset/
│   ├── kertas/
│   ├── logam/
│   └── plastik/
├── features/
│   ├── kertas_features.csv
│   ├── logam_features.csv
│   ├── plastik_features.csv
│   └── dataset_fitur_gabungan.csv
├── models/
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   └── label_encoder.pkl
├── src/
│   ├── extract_kertas.py
│   ├── extract_logam.py
│   ├── extract_plastik.py
│   ├── combine_features.py
│   ├── train_classifiers.py
│   └── predict_image.py
├── test_image/
│   └── sampel1.jpg
└── requirements.txt
```
---

## 🚀 Alur Penggunaan

1. **Ekstraksi Fitur** 🎨🔺🧩  
   Jalankan script ekstraksi fitur untuk setiap kategori:
   ```sh
   python src/extract_kertas.py
   python src/extract_logam.py
   python src/extract_plastik.py
   ```
Hasil: file fitur .csv di folder features/.

2. Gabungkan Fitur 🗃️
Gabungkan semua fitur menjadi satu dataset:
    ```sh
    python src/combine_features.py
Hasil: dataset_fitur_gabungan.csv

3. Training Model 🤖
Latih model KNN & SVM:
     ```sh
    python src/train_classifiers.py
Hasil: file model .pkl di folder models/.

4. Prediksi Gambar Baru 🔍
Prediksi jenis sampah dari gambar:
     ```sh
    python src/predict_image.py
Atur path gambar pada script atau input sesuai kebutuhan.

## 🛠️ Kebutuhan dan Instalasi

### ✅ Kebutuhan:
- Python 3.x
- OpenCV (`opencv-python`)
- scikit-learn
- scikit-image
- numpy

### 📦 Instalasi Dependencies:
Jalankan perintah berikut untuk menginstal semua dependencies:
```sh
pip install -r requirements.txt
```
## 📜 Penjelasan Script

- `extract_kertas.py`, `extract_logam.py`, `extract_plastik.py`  
  Ekstraksi fitur warna, bentuk, dan tekstur dari gambar di masing-masing folder dataset.
```sh
# Ekstraksi fitur
python src/extract_kertas.py
python src/extract_logam.py
python src/extract_plastik.py
```

- `combine_features.py`  
  Menggabungkan semua file fitur menjadi satu dataset gabungan.
```sh
# Gabungkan fitur
python src/combine_features.py
```

- `train_classifiers.py`  
  Melatih model KNN dan SVM dari dataset gabungan, hasilnya file model `.pkl` di folder `models/`.
```sh
# Training model
python src/train_classifiers.py
```
- `predict_image.py`  
  Melakukan prediksi jenis sampah dari gambar baru menggunakan model yang sudah dilatih.
```sh
# Prediksi gambar baru
python src/predict_image.py
```
- `test_image/`  
  Folder berisi contoh gambar untuk pengujian.

---

