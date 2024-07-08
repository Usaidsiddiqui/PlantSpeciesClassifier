import numpy as np
import os
import joblib
import argparse
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
from tkinter import Tk, filedialog

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
        self.var = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.flatten()
    return image

def predict_image(model, le, image_path, pca):
    image = load_image(image_path)
    image = pca.transform(image.reshape(1, -1))  # Transform and reshape for PCA
    pred = model.predict(image)
    label = le.inverse_transform(pred)
    return label[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", help="./Dataset")
    ap.add_argument("-m", "--mode", required=True, choices=['accuracy', 'predict'], help="mode to run the script: 'accuracy' or 'predict'")
    args = vars(ap.parse_args())
    model_trained = True
    pca = PCA(n_components=100, whiten=True, random_state=42)  # Initialize PCA object

    model_path = "gnb_model_manual.pkl"

    if os.path.exists(model_path):
        print("[INFO] loading the saved GNB model...")
        model, le, saved_pca = joblib.load(model_path)
        if saved_pca is not None:
            pca = saved_pca  # Use saved PCA object if available
    else:
        model_trained = False
        print("[INFO] loading images...")
        imagePaths = list(paths.list_images(args["dataset"]))
        sp = SimplePreprocessor.SimplePreprocessor(32, 32)
        sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
        (data, labels) = sdl.load(imagePaths, verbose=500)
        data = data.reshape((data.shape[0], 3072))
        print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

        le = LabelEncoder()
        labels = le.fit_transform(labels)
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

        print("[INFO] training the GNB model...")
        trainX_pca = pca.fit_transform(trainX)  # Fit PCA on training data
        model = GaussianNaiveBayes()
        model.fit(trainX_pca, trainY)

        joblib.dump((model, le, pca), model_path)
        print("[INFO] GNB model saved...")

    if args["mode"] == 'accuracy':

        if not args["dataset"]:
            print("[ERROR] Dataset path is required for accuracy check.")
            return

        if model_trained == True:
            print("[INFO] loading images...")
            imagePaths = list(paths.list_images(args["dataset"]))
            sp = SimplePreprocessor.SimplePreprocessor(32, 32)
            sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
            (data, labels) = sdl.load(imagePaths, verbose=500)
            data = data.reshape((data.shape[0], 3072))
            print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

            le = LabelEncoder()
            labels = le.fit_transform(labels)
            (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

        print("[INFO] evaluating Gaussian Naive Bayes classifier...")
        testX_pca = pca.transform(testX)  # Transform test data using fitted PCA
        predictions = model.predict(testX_pca)
        print(classification_report(testY, predictions, target_names=le.classes_))
    elif args["mode"] == 'predict':
        # Open file dialog to select image
        root = Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename()

        if image_path:
            label = predict_image(model, le, image_path, pca)
            print(f"[INFO] Predicted class: {label}")
        else:
            print("[ERROR] No image selected.")

if __name__ == "__main__":
    main()
