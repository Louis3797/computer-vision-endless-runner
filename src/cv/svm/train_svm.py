import joblib
import numpy as np
import cv2
import os

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
import time
import hog as hog
from PIL import Image, ImageEnhance
from sklearn.decomposition import PCA

# Constants

# n_components = 1000
sample_size = (96, 160)
test_size = 0.3
max_samples = 10000000

# =====================================


def get_image_paths(directory, max_images=10000000000000000):
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more extensions if needed
    image_paths = []

    limit = 0

    # Check if the directory exists
    if os.path.exists(directory) and os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if limit == max_images:
                    return image_paths
                # Check if the file has an image extension
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
                    print(len(image_paths))
                    limit += 1
    else:
        print("Directory doesn't exist or is not a directory.")

    return image_paths


def process_sample(image):
    fd, _ = hog.HOG().compute(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(3,3), sobel=False, visualize=False,
                              normalize_input=True, flatten=True)

    return fd


positive_path = r"/Users/louis/PycharmProjects/cv_project/src/cv/svm/output_96_160"  # This is the path of our positive input dataset
negative_path = r"/Users/louis/Downloads/INRIAPerson/Train/neg"

print("Loading Samples...")
# load image paths
positive_samples = get_image_paths(positive_path)
# positive_samples.extend(get_image_paths("/Users/louis/Downloads/INRIAPerson/Test/pos"))
positive_samples.extend(get_image_paths("/Users/louis/PycharmProjects/ki/cv/new_positive", 500))
negative_samples = get_image_paths(negative_path)
# negative_samples.extend(get_image_paths("/Users/louis/Downloads/INRIAPerson/Test/neg"))
negative_samples.extend(get_image_paths("/Users/louis/PycharmProjects/cv_project/dataset/negative", 500))


size_pos_data, size_neg_data = len(positive_samples), len(negative_samples)

print(f"Amount of positive (with person) data samples: {size_pos_data}")
print(f"Amount of negative (without person) data samples: {size_neg_data}")
print(f"Total amount : {size_pos_data + size_neg_data}")

# holds feature vectors
features = []
labels = []


st = time.time()
print("Process positive samples...")
# Process positive samples - load, resize, convert to greyscale, compute hog features
for idx, path in enumerate(positive_samples):
    if idx == max_samples:
        break

    image = Image.open(path)

    image = image.convert("L")

    image = np.array(image)
    image = cv2.resize(image, sample_size)
    feature_vector = process_sample(image)

    features.append(feature_vector)
    labels.append(1)  # positive samples are labeled as 1

    flipped = cv2.flip(image, 1)

    feature_vector_2 = process_sample(flipped)

    features.append(feature_vector_2)
    labels.append(1)  # positive samples are labeled as 1

print("Finished")

print("Process negative samples...")
# Process negative samples - load, resize, convert to greyscale, compute hog features
for idx, path in enumerate(negative_samples):

    if idx == max_samples:
        break
    image = Image.open(path)
    image = image.convert("L")

    image = np.array(image)
    image = cv2.resize(image, sample_size)
    feature_vector = process_sample(image)

    features.append(feature_vector)
    labels.append(0)  # negative samples are labeled as 0

    flipped = cv2.flip(image, 1)

    feature_vector_2 = process_sample(flipped)

    features.append(feature_vector_2)
    labels.append(0)  # positive samples are labeled as 1


print("Finished")

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# # ===============
#
# positive_training_path = r"/Users/louis/PycharmProjects/ki/cv/positive"  # This is the path of our positive input dataset
# # positive_training_path = r"/Users/louis/Downloads/INRIAPerson/Test/pos"  # This is the path of our positive input dataset
# negative_training_path = r"/Users/louis/Downloads/INRIAPerson/Test/neg"
#
# print("Loading extra Training Samples...")
# # load image paths
# positive_training_samples = get_image_paths(positive_training_path)
# negative_training_samples = get_image_paths(negative_training_path)
#
# size_pos_training_data, size_neg_training_data = len(positive_training_samples), len(negative_training_samples)
#
# print(f"Amount of positive (with person) data samples: {size_pos_training_data}")
# print(f"Amount of negative (without person) data samples: {size_neg_training_data}")
# print(f"Total amount : {size_pos_training_data + size_neg_training_data}")
#
# train_features = []
# train_labels = []
#
# print("Process additional training positive samples...")
# # Process positive samples - load, resize, convert to greyscale, compute hog features
# for idx, path in enumerate(positive_training_samples):
#     # if idx == max_samples -1:
#     #     break
#
#     image = Image.open(path)
#     image = image.convert("L")
#     image = np.array(image)
#     feature_vector = process_sample(image)
#
#     train_features.append(feature_vector)
#     train_labels.append(1)  # positive samples are labeled as 1
#
# print("Finished")
#
#
# print("Process additional training negative samples...")
# # Process negative samples - load, resize, convert to greyscale, compute hog features
# for idx, path in enumerate(negative_training_samples):
#
#     # if idx == max_samples -1:
#     #     break
#     image = Image.open(path)
#
#     image = image.convert("L")
#
#     image = np.array(image)
#
#
#     feature_vector = process_sample(image)
#
#     train_features.append(feature_vector)
#     train_labels.append(0)  # negative samples are labeled as 0
#
# print("Finished")
#
#
# train_features = np.array(train_features)
# train_features = train_features.reshape(train_features.shape[0], -1)


features = np.array(features)
features = features.reshape(features.shape[0], -1)  # Reshape to flatten the last dimension

print(features.shape)
# encode labels
print("Endcoding Labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
# train_labels = label_encoder.fit_transform(train_labels)

print("Constructing training/testing split...")
# x: feature vectors
# y: labels

x_train, x_test, y_train, y_test = train_test_split(
    np.array(features), labels, test_size=test_size, random_state=42)

print(f"shape x_train: {x_train.shape}")

print("Training Linear SVM classifier...")
model = SVC(kernel='linear')
model.fit(x_train, y_train)

print("Evaluating classifier on test data ...")

predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

m = confusion_matrix(y_test, predictions)
print(m)

# Save the SVM classifier after PCA
joblib.dump(model, 'svm_detection_inria_with_flipped_and_anno_96_160_p_tt_500.dat')

