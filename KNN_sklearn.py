# from sklearn.neighbors import KNeighborsClassifier
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

# print("[INFO] training k-NN model...")
# model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
# model.fit(trainX, trainY)
    
# print("[INFO] evaluating k-NN model...")
# predictions = model.predict(testX)
# print(classification_report(testY, predictions, target_names=le.classes_))


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import cv2
from tkinter import Tk, filedialog
import joblib
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--mode", required=True, choices=['accuracy', 'predict'], help="mode to run the script: 'accuracy' or 'predict'")
    ap.add_argument("-k", "--neighbors", type=int, default=15, help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    
    model_path = "knn_model_sklearn.pkl"
    model_trained = False

    if os.path.exists(model_path):
        print("[INFO] loading the saved k-NN model...")
        model, le, trainX, trainY = joblib.load(model_path)
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
        
        print("[INFO] training k-NN model...")
        model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
        model.fit(trainX, trainY)
        
        # Save the trained model
        joblib.dump((model, le, trainX, trainY), model_path)
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

        print("[INFO] evaluating k-NN model...")
        predictions = model.predict(testX)
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
            
            # Make prediction
            prediction = model.predict([image])
            predicted_label = le.inverse_transform(prediction)[0]
            print(f"[INFO] Predicted class: {predicted_label}")
        else:
            print("[ERROR] No image selected.")

if __name__ == "__main__":
    main()
