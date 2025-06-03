import os
import cv2
import numpy as np
import csv
from skimage.feature import graycomatrix, graycoprops

# --- Fitur Warna ---
def extract_color_features(image):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features

# --- Fitur Bentuk (Hu Moments) ---
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

# --- Fitur Tekstur (GLCM) ---
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

def process_kertas(folder_path, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        # warna 32 bins x 3 channel
        header += [f'color_hist_{i}' for i in range(32*3)]
        # bentuk Hu Moments 7
        header += [f'hu_moment_{i+1}' for i in range(7)]
        # tekstur 6
        header += ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
        header.append('label')
        writer.writerow(header)

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                path = os.path.join(folder_path, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                c = extract_color_features(img)
                s = extract_shape_features(img)
                t = extract_texture_features(img)
                row = c + s + t + ['kertas']
                writer.writerow(row)

if __name__ == "__main__":
    dataset_folder = os.path.join('..','dataset','kertas')
    output_file = os.path.join('..','features','kertas_features.csv')
    process_kertas(dataset_folder, output_file)
