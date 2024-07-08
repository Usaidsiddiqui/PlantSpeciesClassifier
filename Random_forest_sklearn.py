# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# import os
# import joblib
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# args = vars(ap.parse_args())

# model_path = "random_forest_sklearn.pkl"

# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))

# # Initialize the image preprocessor, load the dataset from disk,
# # and reshape the data matrix
# sp = SimplePreprocessor.SimplePreprocessor(32, 32)
# sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
# (data, labels) = sdl.load(imagePaths, verbose=500)
# data = data.reshape((data.shape[0], 3072))

# # Show some information on memory consumption of the images
# print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# # Encode labels
# le = LabelEncoder()
# labels = le.fit_transform(labels)

# # Partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# # Check if the model is already saved
# if os.path.exists(model_path):
#     print("[INFO] loading the saved Decision Tree model...")
#     model, le = joblib.load(model_path)
# else:
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(trainX, trainY)
    
#     # Save the trained model and the label encoder
#     joblib.dump((model, le), model_path)
#     print("[INFO] Decision Tree model saved...")

# # Evaluate the model
# print("[INFO] evaluating Decision Tree model...")
# predictions = model.predict(testX)
# print(classification_report(testY, predictions, target_names=le.classes_))






import numpy as np
import os
import joblib
import argparse
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
from tkinter import Tk, filedialog

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    image = image.flatten()
    return image

def predict_image(model, le, image_path):
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    label = le.inverse_transform(pred)
    return label[0]

def main():
    # Argument parser setup
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", help="path to input dataset")
    ap.add_argument("-m", "--mode", required=True, choices=['accuracy', 'predict'], help="mode to run the script: 'accuracy' or 'predict'")
    args = vars(ap.parse_args())
    model_trained = True

    # Load the model and label encoder
    model_path = "random_forest_sklearn.pkl"
    if os.path.exists(model_path):
        print("[INFO] loading the saved RandomForest model...")
        model, le = joblib.load(model_path)
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

        print("[INFO] training the RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(trainX, trainY)

        print("[INFO] saving the RandomForest model...")
        joblib.dump((model, le), model_path)

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

        print("[INFO] evaluating RandomForest model...")
        predictions = model.predict(testX)
        print(classification_report(testY, predictions, target_names=le.classes_))

    elif args["mode"] == 'predict':
        # Open file dialog to select image
        root = Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename()

        if image_path:
            label = predict_image(model, le, image_path)
            print(f"[INFO] Predicted class: {label}")
        else:
            print("[ERROR] No image selected.")

if __name__ == "__main__":
    main()
