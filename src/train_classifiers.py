import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import joblib

def load_dataset(csv_path):
    X = []
    y = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            X.append([float(val) for val in row[:-1]])
            y.append(row[-1])
    return np.array(X), np.array(y)

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

    # SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    # Simpan model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(knn, "../models/knn_model.pkl")
    joblib.dump(svm, "../models/svm_model.pkl")
    print("Model KNN dan SVM telah disimpan di folder models.")

    return svm, knn

if __name__ == "__main__":
    csv_file = os.path.join('..','features','dataset_fitur_gabungan.csv')
    X, y = load_dataset(csv_file)
    y_encoded, label_encoder = encode_labels(y)
    svm, knn = train_and_evaluate(X, y_encoded)
    # Simpan label encoder
    joblib.dump(label_encoder, "../models/label_encoder.pkl")
    print("Label encoder telah disimpan di folder models.")