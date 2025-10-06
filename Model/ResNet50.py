from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet101V2
from keras.applications import ResNet101
from keras.applications import ResNet152
resnet50 = ResNet50(weights='imagenet', include_top=False)
resnet101 = ResNet101(weights='imagenet', include_top=False)
resnet101v2 = ResNet101V2(weights='imagenet', include_top=False)
resnet152 = ResNet152(weights='imagenet', include_top=False)
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os
from keras.applications.vgg16 import preprocess_input
from google.colab import drive
drive.mount('/content/drive')


features_representation_1 = resnet_features.flatten()
features_representation_2 = resnet_features.squeeze()

print ("Shape 1: ", features_representation_1.shape)

print ("Shape 2: ", features_representation_2.shape)

#new
import os
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import numpy as np
import pandas as pd

def _get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming ResNet-50 input size
    print("path: ",img_path)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resnet_features = resnet101.predict(img_data)
    return resnet_features

def extract_features_from_directory(directory, class_label):
    features_list = []
    image_paths = []
    labels = []  # List to store class labels

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            features = _get_features(img_path)
            features_list.append(features.flatten())  # Flatten the features to store in a list
            image_paths.append(img_path)
            labels.append(class_label)  # Add class label for this image

    return features_list, image_paths, labels

# Directory containing images and its corresponding class label
directory_eczema = '/content/drive/MyDrive/Senior Project/imbalanced_non_segmented_images/Eczema images'
directory_non_eczema = '/content/drive/MyDrive/Senior Project/imbalanced_non_segmented_images/Non eczema images'

# Extract features and labels for eczema images
features_list_eczema, image_paths_eczema, labels_eczema = extract_features_from_directory(directory_eczema, class_label="eczema")

# Extract features and labels for non-eczema images
features_list_non_eczema, image_paths_non_eczema, labels_non_eczema = extract_features_from_directory(directory_non_eczema, class_label="non-eczema")
# Combine features and labels from both classes
features_list_combined = features_list_eczema + features_list_non_eczema
image_paths_combined = image_paths_eczema + image_paths_non_eczema
labels_combined = labels_eczema + labels_non_eczema

# Convert the combined list of features to a DataFrame for easier handling
features_df = pd.DataFrame(features_list_combined)

# Add image paths and class labels to the DataFrame
#features_df['Image Path'] = image_paths_combined
features_df['Class Label'] = labels_combined

# Save features, image paths, and class labels to a CSV file
features_df.to_csv("/content/drive/MyDrive/Senior Project/eczema_non_eczema_resnet101_features_balanced_segmented.csv", index=False)

data = '/content/drive/MyDrive/Senior Project/extracted_features2_vgg19.csv'

df = pd.read_csv(data)

X = df.drop(['label'], axis=1)
y = df['label']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #check splitting in past papers

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import required libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Instantiate classifier with rbf kernel and C=1.0
svc = SVC()

# Fit classifier to training set
svc.fit(X_train, y_train)

# Make predictions on test set
y_pred_test = svc.predict(X_test)

# Compute and print accuracy score
accuracy = accuracy_score(y_test, y_pred_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy))

# Compute other performance metrics
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Print other metrics
print('Precision: {0:0.4f}'.format(precision))
print('Recall: {0:0.4f}'.format(recall))
print('F1-score: {0:0.4f}'.format(f1))
print('Confusion Matrix:')
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=conf_matrix, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


