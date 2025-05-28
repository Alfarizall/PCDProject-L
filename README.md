# 🗑️ Sistem Klasifikasi Sampah dengan CNN

Proyek ini menggunakan **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar sampah ke dalam tiga kategori: **organik, daur ulang, dan berbahaya**.

## 📌 Fitur

- Klasifikasi gambar sampah menggunakan CNN
- Preprocessing dan augmentasi dataset
- Prediksi real-time (opsional via webcam atau gambar)
- UI sederhana (opsional dengan Streamlit atau Flask)

## 🧠 Metode

Model CNN dilatih menggunakan dataset gambar yang telah dilabeli. Kategori klasifikasi:
- Sampah Organik
- Sampah Daur Ulang
- Sampah Berbahaya

## 🗃️ Dataset

- Sumber: Dataset publik atau buatan sendiri
- Format: Gambar JPEG/PNG, disusun dalam folder sesuai kelas

## 🚀 Cara Menjalankan

```bash
git clone https://github.com/username/waste-cnn.git
cd waste-cnn
pip install -r requirements.txt
python train.py        # Melatih model
python predict.py      # Prediksi satu gambar
