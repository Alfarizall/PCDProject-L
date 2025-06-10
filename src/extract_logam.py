import os
import cv2
import numpy as np
import csv
from skimage.feature import graycomatrix, graycoprops

def extract_color_features(image):
    image = cv2.resize(image, (128, 128))
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [32], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features

def extract_shape_features(image):
    image = cv2.resize(image, (128, 128))
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
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    ASM = graycoprops(glcm, 'ASM')[0,0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

def process_logam(folder_path, output_csv, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        header += [f'color_hist_{i}' for i in range(32*3)]
        header += [f'hu_moment_{i+1}' for i in range(7)]
        header += ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
        header.append('label')
        writer.writerow(header)

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder_path, file)
                img = cv2.imread(path)
                if img is None:
                    continue

                c = extract_color_features(img)
                s = extract_shape_features(img)
                t = extract_texture_features(img)
                row = c + s + t + ['logam']
                writer.writerow(row)

                # Simpan output visualisasi citra
                basename = os.path.splitext(file)[0]
                
                # Simpan salinan citra yang sudah di-resize
                original_resized = cv2.resize(img, (128, 128))
                cv2.imwrite(os.path.join(output_image_dir, f"{basename}_original.jpg"), original_resized)

                gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shape_img = original_resized.copy()
                cv2.drawContours(shape_img, contours, -1, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(output_image_dir, f"{basename}_shape.jpg"), shape_img)

                # Texture (grayscale only)
                cv2.imwrite(os.path.join(output_image_dir, f"{basename}_gray.jpg"), gray)

if __name__ == "__main__":
    dataset_folder = os.path.join('..', 'dataset', 'logam')
    output_file = os.path.join('..', 'features', 'logam_features.csv')
    output_image_dir = os.path.join('..', 'output_samples', 'logam')
    process_logam(dataset_folder, output_file, output_image_dir)