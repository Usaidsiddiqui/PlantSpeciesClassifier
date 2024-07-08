import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#--------------------------------------------------------------------------------------------------------
# Define the path to your dataset
dataset_path = 'Dataset'
#getting categories
categories = os.listdir(dataset_path)
#--------------------------------------------------------------------------------------------------------






#--------------------------------------------------------------------------------------------------------
#plotting imbalance in classes
plot_data={}
for folder_dir in os.listdir("dataset"):
    class_name=folder_dir
    no_of_images=len(os.listdir(f"dataset/{folder_dir}"))
    plot_data.update({class_name:no_of_images})
print(plot_data)
modified_dict = {}
for key, value in plot_data.items():
    modified_key = key.replace('___healthy', '')
    modified_dict[modified_key] = value

print(modified_dict)
plt.bar(modified_dict.keys(),plot_data.values(),width=0.3)
plt.show()
#--------------------------------------------------------------------------------------------------------







#--------------------------------------------------------------------------------------------------------
# plotting  sample images from each category
sample_images = []
for category in categories:
    category_path = os.path.join(dataset_path, category)
    sample_image_path = os.path.join(category_path, os.listdir(category_path)[0])
    sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    sample_images.append((category, sample_image))
plt.figure(figsize=(15, 10))
sample_images=sample_images[0:5]
for i, (category, image) in enumerate(sample_images):
    plt.subplot(1, len(sample_images), i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(category)
    plt.axis('off')
plt.show()
#--------------------------------------------------------------------------------------------------------








#--------------------------------------------------------------------------------------------------------
#image statistics
image_shapes = []
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_shapes.append(image.shape)

image_shapes = pd.DataFrame(image_shapes, columns=['Height', 'Width'])
print(image_shapes.describe())
#--------------------------------------------------------------------------------------------------------





# import numpy as np
# pixel_values = []

# for category in categories:
#     category_path = os.path.join(dataset_path, category)
#     for filename in os.listdir(category_path):
#         image_path = os.path.join(category_path, filename)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is not None:
#             pixel_values.extend(image.flatten())

# pixel_values = np.array(pixel_values)
# print(f'Mean pixel value: {np.mean(pixel_values)}')
# print(f'Standard deviation of pixel values: {np.std(pixel_values)}')



# plt.figure(figsize=(10, 5))
# plt.hist(pixel_values, bins=50, color='gray')
# plt.title('Distribution of Pixel Values')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()
