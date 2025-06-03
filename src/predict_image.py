import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# --- Fitur Ekstraksi (samakan dengan training) ---
def extract_color_features(image):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features

def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0]*7
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)
    return hu.tolist()

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    ASM = graycoprops(glcm, 'ASM')[0,0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

def extract_all_features(image):
    c = extract_color_features(image)
    s = extract_shape_features(image)
    t = extract_texture_features(image)
    return np.array(c + s + t).reshape(1, -1)

# --- Prediksi ---
# ...existing code...

def predict_images_in_folder(folder_path, model_path, label_encoder_path):
    model = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gambar {file} tidak bisa dibuka.")
                continue
            features = extract_all_features(img)
            pred = model.predict(features)
            label = le.inverse_transform(pred)[0]
            print(f"Gambar '{file}' diprediksi sebagai: {label}")

if __name__ == "__main__":
    # Ganti path di bawah sesuai kebutuhan
    folder_path = "../test_image"  # folder berisi banyak gambar
    model_path = "../models/svm_model.pkl"
    label_encoder_path = "../models/label_encoder.pkl"
    predict_images_in_folder(folder_path, model_path, label_encoder_path)
