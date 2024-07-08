# import numpy as np
# import os
# import joblib
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import SimplePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from imutils import paths
# from sklearn.tree import DecisionTreeClassifier
# import argparse

# # Argument parser setup
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# args = vars(ap.parse_args())

# # RandomForest class
# class RandomForest:
#     def __init__(self, n_trees=10, max_depth=100):
#         self.n_trees = n_trees
#         self.max_depth = max_depth
#         self.trees = []

#     def fit(self, X, y):
#         self.trees = []
#         for _ in range(self.n_trees):
#             idxs = np.random.choice(len(X), len(X), replace=True)
#             print("start")
#             tree = DecisionTreeClassifier(max_depth=self.max_depth)
#             tree.fit(X[idxs], y[idxs])
#             self.trees.append(tree)

#     def predict(self, X):
#         tree_preds = np.array([tree.predict(X) for tree in self.trees])
#         tree_preds = np.swapaxes(tree_preds, 0, 1)
#         return np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds])

# # Loading images
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

# model_path = "random_forest_manual.pkl"

# if os.path.exists(model_path):
#     print("[INFO] loading the saved RandomForest model...")
#     model, le = joblib.load(model_path)
# else:
#     print("[INFO] training the RandomForest model...")
#     model = RandomForest(n_trees=10, max_depth=100)
#     model.fit(trainX, trainY)
#     joblib.dump((model, le), model_path)
#     print("[INFO] RandomForest model saved...")

# print("[INFO] evaluating RandomForest model...")
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
from sklearn.tree import DecisionTreeClassifier
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
from tkinter import Tk, filedialog

# RandomForest class
class RandomForest:
    def __init__(self, n_trees=10, max_depth=100):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            print("start")
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds])

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
    model_path = "random_forest_manual.pkl"
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
        model = RandomForest(n_trees=10, max_depth=100)
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
