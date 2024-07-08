# import the necessary packages
import numpy as np
import cv2
import os
class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors
		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []
	def load(self, imagePaths, verbose=-1):
		print(imagePaths[0])
		data = []
		labels = []
		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
			# print(label)
			if self.preprocessors is not None:
				for p in self.preprocessors:
					image = p.preprocess(image)
				
			data.append(image)
			labels.append(label)
			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))
		return (np.array(data), np.array(labels))
	
	def load_single_image(self, imagePath):
		data = []
		# loop over the input images
		# for (i, imagePath) in enumerate(imagePaths):
		image = cv2.imread(imagePath)

			# print(label)
		if self.preprocessors is not None:
			for p in self.preprocessors:
				print("preprocessing")
				image = p.preprocess(image)
				
		data.append(image)
			# show an update every `verbose` images
		
		return (np.array(data))
	   