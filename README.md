# ♻️ PCDProject-L

Klasifikasi Sampah Otomatis 🗑️ menggunakan Ekstraksi Fitur Citra dan Machine Learning (KNN & SVM)

---

## 📁 Struktur Folder
dataset/ kertas/ logam/ plastik/ features/ kertas_features.csv logam_features.csv plastik_features.csv dataset_fitur_gabungan.csv models/ knn_model.pkl svm_model.pkl label_encoder.pkl src/ extract_kertas.py extract_logam.py extract_plastik.py combine_features.py train_classifiers.py predict_image.py test_image/ sampel1.jpg

---

## 🚀 Alur Penggunaan

1. **Ekstraksi Fitur** 🎨🔺🧩  
   Jalankan script ekstraksi fitur untuk setiap kategori:
   ```sh
   python src/extract_kertas.py
   python src/extract_logam.py
   python src/extract_plastik.py

Hasil: file fitur .csv di folder features/.

2. Gabungkan Fitur 🗃️
Gabungkan semua fitur menjadi satu dataset:
python src/combine_features.py
Hasil: dataset_fitur_gabungan.csv

3. Training Model 🤖
Latih model KNN & SVM:
python src/train_classifiers.py
Hasil: file model .pkl di folder models/.

4. Prediksi Gambar Baru 🔍
Prediksi jenis sampah dari gambar:
python src/predict_image.py
Atur path gambar pada script atau input sesuai kebutuhan.

🛠️ Kebutuhan
Python 3.x
OpenCV (opencv-python)
scikit-learn
scikit-image
numpy
Install dependencies:
pip install -r requirements.txt

📜 Penjelasan Script
extract_kertas.py, extract_logam.py, extract_plastik.py
Ekstraksi fitur warna, bentuk, dan tekstur dari gambar di masing-masing folder dataset.
combine_features.py
Menggabungkan semua file fitur menjadi satu dataset gabungan.
train_classifiers.py
Melatih model KNN dan SVM dari dataset gabungan.
predict_image.py
Melakukan prediksi jenis sampah dari gambar baru.
test_image/
Contoh gambar untuk pengujian.
