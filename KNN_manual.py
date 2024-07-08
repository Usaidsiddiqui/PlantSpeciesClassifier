# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,help="./Dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=15,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))

# def knn_predict(X_train, y_train, X_test, k=3):
#     predictions = []
#     for test_point in X_test:
#         distances = []
#         for i in range(len(X_train)):
#             distance = euclidean_distance(test_point, X_train[i])
#             distances.append((distance, y_train[i]))
#         distances.sort(key=lambda x: x[0])
#         k_nearest_neighbors = [distances[i][1] for i in range(k)]
#         prediction = max(set(k_nearest_neighbors), key=k_nearest_neighbors.count)
#         predictions.append(prediction)
#     return predictions


# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))
# # initialize the image preprocessor, load the dataset from disk,
# # and reshape the data matrix
# sp = SimplePreprocessor.SimplePreprocessor(32, 32)
# sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
# (data, labels) = sdl.load(imagePaths, verbose=500)
# data = data.reshape((data.shape[0], 3072))
# # show some information on memory consumption of the images
# print("[INFO] features matrix: {:.1f}MB".format(
# 	data.nbytes / (1024 * 1024.0)))

# le = LabelEncoder()
# labels = le.fit_transform(labels)
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.25, random_state=42)

# print("[INFO] evaluating k-NN classifier...")
# predictions = knn_predict(trainX, trainY, testX, k=args["neighbors"])
# print(classification_report(testY, predictions,
# 	target_names=le.classes_))

# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
# import argparse

# class KNNModel:
#     def __init__(self, n_neighbors=15, batch_size=50):
#         self.n_neighbors = n_neighbors
#         self.batch_size = batch_size
    
#     def fit(self, X, y):
#         self.trainX = X
#         self.trainY = y
    
#     def predict(self, X):
#         n_samples = X.shape[0]
#         predictions = np.zeros(n_samples)
        
#         for i in range(0, n_samples, self.batch_size):
#             end = min(i + self.batch_size, n_samples)
#             batch_predictions = self._predict_batch(X[i:end])
#             predictions[i:end] = batch_predictions
        
#         return predictions
    
#     def _predict_batch(self, X_batch):
#         n_train_samples = self.trainX.shape[0]
#         batch_predictions = np.zeros(X_batch.shape[0])
        
#         for i in range(0, n_train_samples, self.batch_size):
#             train_end = min(i + self.batch_size, n_train_samples)
#             distances = np.linalg.norm(self.trainX[i:train_end, np.newaxis] - X_batch, axis=2)
#             for j in range(X_batch.shape[0]):
#                 k_indices = np.argsort(distances[:, j])[:self.n_neighbors]
#                 k_nearest_labels = self.trainY[i:train_end][k_indices]
#                 batch_predictions[j] = np.bincount(k_nearest_labels).argmax()
        
#         return batch_predictions

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=15, help="# of nearest neighbors for classification")
# args = vars(ap.parse_args())

# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))
# sp = SimplePreprocessor.SimplePreprocessor(32, 32)
# sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
# (data, labels) = sdl.load(imagePaths, verbose=500)
# data = data.reshape((data.shape[0], 3072))
# print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# le = LabelEncoder()
# labels = le.fit_transform(labels)
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# print("[INFO] training k-NN model...")
# model = KNNModel(n_neighbors=args["neighbors"])
# model.fit(trainX, trainY)

# print("[INFO] evaluating k-NN model...")
# predictions = model.predict(testX)
# print(classification_report(testY, predictions, target_names=le.classes_))
# print(f"Accuracy: {accuracy_score(testY, predictions):.2f}")


# import cv2
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
# import argparse

# class KNNClassifier:
#     def __init__(self, k=3):
#         self.k = k

#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train

#     def predict(self, X_test):
#         predictions = []
#         for test_point in X_test:
#             distances = [euclidean_distance(test_point, x) for x in self.X_train]
#             nearest_neighbors = np.argsort(distances)[:self.k]
#             k_nearest_labels = [self.y_train[i] for i in nearest_neighbors]
#             prediction = np.argmax(np.bincount(k_nearest_labels))
#             predictions.append(prediction)
#         return predictions

# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
#     ap.add_argument("-m", "--mode", required=True, choices=['accuracy', 'predict'], help="mode to run the script: 'accuracy' or 'predict'")
#     ap.add_argument("-k", "--neighbors", type=int, default=15, help="# of nearest neighbors for classification")
#     args = vars(ap.parse_args())

#     print("[INFO] loading images...")
#     imagePaths = list(paths.list_images(args["dataset"]))
    
#     # initialize the image preprocessor, load the dataset from disk,
#     # and reshape the data matrix
#     sp = SimplePreprocessor.SimplePreprocessor(32, 32)
#     sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
#     (data, labels) = sdl.load(imagePaths, verbose=500)
#     data = data.reshape((data.shape[0], 3072))
    
#     # show some information on memory consumption of the images
#     print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))
    
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)
    
#     # partition the data into training and testing splits using 75% of
#     # the data for training and the remaining 25% for testing
#     trainX, testX, trainY, testY = train_test_split(data, labels,
#         test_size=0.25, random_state=42)
    
#     if args["mode"] == 'accuracy':
#         print("[INFO] evaluating k-NN classifier...")
#         knn = KNNClassifier(k=args["neighbors"])
#         knn.fit(trainX, trainY)
#         predictions = knn.predict(testX)
#         print(classification_report(testY, predictions, target_names=le.classes_))
    
#     elif args["mode"] == 'predict':
#         # Perform prediction on a single image
#         from tkinter import Tk, filedialog
#         root = Tk()
#         root.withdraw()
#         image_path = filedialog.askopenfilename()
        
#         if image_path:
#             # Load the image and preprocess it
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, (32, 32))
#             image = image.flatten()
            
#             # Initialize and fit PCA if needed (you can reuse the PCA code from your previous example)
#             pca = PCA(n_components=100, whiten=True, random_state=42)
#             trainX_pca = pca.fit_transform(trainX)
            
#             # Initialize the k-NN classifier and fit it with training data
#             knn = KNNClassifier(k=args["neighbors"])
#             knn.fit(trainX_pca, trainY)
            
#             # Transform the image data using PCA and predict the class
#             image_pca = pca.transform(image.reshape(1, -1))
#             prediction = knn.predict(image_pca)
#             predicted_label = le.inverse_transform(prediction)[0]
#             print(f"[INFO] Predicted class: {predicted_label}")
#         else:
#             print("[ERROR] No image selected.")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import joblib
import os
from tkinter import Tk, filedialog

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i in range(len(self.X_train)):
                distance = self.euclidean_distance(test_point, self.X_train[i])
                distances.append((distance, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = [distances[i][1] for i in range(self.k)]
            prediction = max(set(k_nearest_neighbors), key=k_nearest_neighbors.count)
            predictions.append(prediction)
        return predictions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--mode", required=True, choices=['accuracy', 'predict'], help="mode to run the script: 'accuracy' or 'predict'")
    ap.add_argument("-k", "--neighbors", type=int, default=15, help="# of nearest neighbors for classification")
    args = vars(ap.parse_args())

    model_path = "knn_model_manual.pkl"
    model_trained = False

    if os.path.exists(model_path):
        print("[INFO] loading the saved k-NN model...")
        knn, le, pca, trainX, trainY = joblib.load(model_path)
        model_trained = True
    else:
        print("[INFO] loading images...")
        imagePaths = list(paths.list_images(args["dataset"]))
        sp = SimplePreprocessor.SimplePreprocessor(32, 32)
        sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
        (data, labels) = sdl.load(imagePaths, verbose=500)
        data = data.reshape((data.shape[0], 3072))
    
        # show some information on memory consumption of the images
        print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))
    
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
        
        # Initialize and fit PCA on training data
        pca = PCA(n_components=100, whiten=True, random_state=42)
        trainX_pca = pca.fit_transform(trainX)
        
        print("[INFO] training k-NN model...")
        knn = KNNClassifier(k=args["neighbors"])
        knn.fit(trainX_pca, trainY)
        
        # Save the trained model
        joblib.dump((knn, le, pca, trainX, trainY), model_path)
        print("[INFO] k-NN model saved...")

    if args["mode"] == 'accuracy':
        if model_trained:
            print("[INFO] loading images...")
            imagePaths = list(paths.list_images(args["dataset"]))
            sp = SimplePreprocessor.SimplePreprocessor(32, 32)
            sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
            (data, labels) = sdl.load(imagePaths, verbose=500)
            data = data.reshape((data.shape[0], 3072))
        
            # show some information on memory consumption of the images
            print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))
        
            le = LabelEncoder()
            labels = le.fit_transform(labels)
        
            # partition the data into training and testing splits using 75% of
            # the data for training and the remaining 25% for testing
            trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

        testX_pca = pca.transform(testX)
        print("[INFO] evaluating k-NN classifier...")
        predictions = knn.predict(testX_pca)
        print(classification_report(testY, predictions, target_names=le.classes_))
    
    elif args["mode"] == 'predict':
        # Perform prediction on a single image
        root = Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename()
        
        if image_path:
            # Load the image and preprocess it
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))
            image = image.flatten()
            
            # Transform the image data using PCA and predict the class
            image_pca = pca.transform(image.reshape(1, -1))
            prediction = knn.predict(image_pca)
            predicted_label = le.inverse_transform(prediction)[0]
            print(f"[INFO] Predicted class: {predicted_label}")
        else:
            print("[ERROR] No image selected.")

if __name__ == "__main__":
    main()
