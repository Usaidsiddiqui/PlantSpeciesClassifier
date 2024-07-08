# import pandas as pd
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,help="./Dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=15,
# 	help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# 	help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

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
        print(self.classes)
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
    # def fit(self, X, y):
    #     self.classes = np.unique(y)
    #     self.mean = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
    #     self.var = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
    #     self.priors = np.zeros(len(self.classes), dtype=np.float64)
        
    #     for idx, c in enumerate(self.classes):
    #         X_c = X[y == c]
    #         self.mean[idx, :] = X_c.mean(axis=0)
    #         self.var[idx, :] = X_c.var(axis=0)
    #         self.priors[idx] = X_c.shape[0] / float(X.shape[0])
    
    # def predict(self, X):
    #     print(len(self.classes))
    #     y_pred = [self._predict(x) for x in X]
    #     return np.array(y_pred)
    
    # def _predict(self, x):
    #     posteriors = []
        
    #     for idx, c in enumerate(self.classes):
    #         prior = np.log(self.priors[idx])
    #         posterior = np.sum(np.log(self._pdf(idx, x)))
    #         posterior = prior + posterior
    #         posteriors.append(posterior)
        
    #     return self.classes[np.argmax(posteriors)]
    
    # def _pdf(self, class_idx, x):
    #     mean = self.mean[class_idx]
    #     var = self.var[class_idx]
    #     numerator = np.exp(- (x - mean) ** 2 / (2 * var))
    #     denominator = np.sqrt(2 * np.pi * var)
    #     return numerator / denominator

# def main_function(to_predict_image_path):
#     model_path = "gnb_model.pkl"
#     print("[INFO] loading images...")
#     imagePaths = list(paths.list_images("Dataset"))
#     # print(imagePaths)
#     # initialize the image preprocessor, load the dataset from disk,
# # and reshape the data matrix
#     sp = SimplePreprocessor.SimplePreprocessor(32, 32)
#     sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
#     # (data, labels) = sdl.load(imagePaths, verbose=500)
#     # data = data.reshape((data.shape[0], 3072))
#     # # show some information on memory consumption of the images
#     # print("[INFO] features matrix: {:.1f}MB".format(
# 	# data.nbytes / (1024 * 1024.0)))
#     (data, labels) = sdl.load(imagePaths, verbose=500)
#     # print(data)
#     print(len(data))
#     data = data.reshape((data.shape[0], 3072))
#     print(data[0])
#     # show some information on memory consumption of the images
#     print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
#     (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.25, random_state=42)
# # Perform PCA for dimensionality reduction
#     pca = PCA(n_components=100, whiten=True, random_state=42)
#     trainX_pca = pca.fit_transform(trainX)
#     testX_pca = pca.transform(testX)
#     print(len(testX_pca))
#     print(testX_pca[0])

# # Check if the model is already saved
#     if os.path.exists(model_path):
#         print("[INFO] loading the saved GNB model...")
#         model, le = joblib.load(model_path)
#     else:
#         print("[INFO] training the GNB model...")
#         model = GaussianNaiveBayes()
#         model.fit(trainX_pca, trainY)
#         joblib.dump((model, le), model_path)
#         print("[INFO] GNB model saved...")

#     print("[INFO] evaluating Guassian Naive Bayes classifier...")
#     print("here is testx")
#     print(len(testX_pca))
#     print("below are the predictons ")
#     data1 = sdl.load_single_image(to_predict_image_path)
#     data1 = data1.reshape((data1.shape[0], 3072))
#     data1=data1.flatten().reshape(1, -1)
#     print("single mage data ")
#     print(f"this is data  :{len(data1)}")
#     # print(len(data1[0]))
#     print(data1[0])
#     test_this = pca.transform(data1)
#     predictions = model.predict(testX_pca)
#     list_of_classes=["Apple","blueberry","cherry","corn","grape","peach","soybean","strawbery","tomato"]
#     predictions2 = model.predict(test_this)
#     print(list_of_classes[predictions2[0]])
    

#     # predictions = knn_predict(trainX, trainY, testX, k=args["neighbors"])
#     print(classification_report(testY, predictions,target_names=le.classes_))
# main_function("tomata.jpg")


def main_function(to_predict_image_path, model_name="gnb"):
    if(model_name=="gnb"):
        model_path = "gnb_model.pkl"
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images("Dataset"))
    # print(imagePaths)
    # initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
    sp = SimplePreprocessor.SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
    # (data, labels) = sdl.load(imagePaths, verbose=500)
    # data = data.reshape((data.shape[0], 3072))
    # # show some information on memory consumption of the images
    # print("[INFO] features matrix: {:.1f}MB".format(
	# data.nbytes / (1024 * 1024.0)))
    (data, labels) = sdl.load(imagePaths, verbose=500)
    # print(data)
    print(len(data))
    data = data.reshape((data.shape[0], 3072))
    print(data[0])
    # show some information on memory consumption of the images
    print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

    le = LabelEncoder()
    labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# Perform PCA for dimensionality reduction
    pca = PCA(n_components=100, whiten=True, random_state=42)
    trainX_pca = pca.fit_transform(trainX)
    testX_pca = pca.transform(testX)
    print(len(testX_pca))
    print(testX_pca[0])

# Check if the model is already saved
    if os.path.exists(model_path):
        print("[INFO] loading the saved GNB model...")
        model, le = joblib.load(model_path)
    else:
        print("[INFO] training the GNB model...")
        model = GaussianNaiveBayes()
        model.fit(trainX_pca, trainY)
        joblib.dump((model, le), model_path)
        print("[INFO] GNB model saved...")

    print("[INFO] evaluating Guassian Naive Bayes classifier...")
    print("here is testx")
    print(len(testX_pca))
    print("below are the predictons ")
    data1 = sdl.load_single_image(to_predict_image_path)
    data1 = data1.reshape((data1.shape[0], 3072))
    data1=data1.flatten().reshape(1, -1)
    print("single mage data ")
    print(f"this is data  :{len(data1)}")
    # print(len(data1[0]))
    print(data1[0])
    test_this = pca.transform(data1)
    predictions = model.predict(testX_pca)
    list_of_classes=["Apple","blueberry","cherry","corn","grape","peach","rasberry","soyabean","strawberry","tomato"]
    predictions2 = model.predict(test_this)
    print(classification_report(testY, predictions,target_names=le.classes_))
    return list_of_classes[predictions2[0]]
    

    # predictions = knn_predict(trainX, trainY, testX, k=args["neighbors"])
    print(classification_report(testY, predictions,target_names=le.classes_))
    # model_path = "gnb_model.pkl"
    # sp = SimplePreprocessor.SimplePreprocessor(32, 32)
    # sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])

    # # Load and preprocess the image
    # data1 = sdl.load_single_image(to_predict_image_path)
    # data1 = data1.reshape((data1.shape[0], 3072)).flatten().reshape(1, -1)
    
    # # Load the model and PCA
    # if model_name == "gnb":
    #     model_path = "gnb_model.pkl"
    # # Add logic for other models here...

    # model, le= joblib.load(model_path)

    # # Transform the image using PCA
    # test_this = pca.transform(data1)
    
    # # Make prediction
    # prediction = model.predict(test_this)
    # list_of_classes = ["Apple", "blueberry", "cherry", "corn", "grape", "peach", "soybean", "strawberry", "tomato"]
    # return list_of_classes[prediction[0]]