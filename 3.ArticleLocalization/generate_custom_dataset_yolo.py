import os
import shutil
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path to the dataset CSV file
dataset_file = 'stl_dataset_for_yolo.csv'

# Create folders for images and labels
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/val', exist_ok=True)
os.makedirs('labels/train', exist_ok=True)
os.makedirs('labels/val', exist_ok=True)

# Read the dataset file
with open(dataset_file, 'r') as file:
    lines = file.readlines()
    header = lines[0].strip().split(',')
    data_lines = [line.strip().split(',') for line in lines[1:]]

# Extract labels and perform label encoding
labels = [line[header.index('category')] for line in data_lines]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into train and validation sets with stratified sampling
train_data, val_data, train_labels, val_labels = train_test_split(
    data_lines, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Function to copy and rename images and create label files
def process_data(data, labels, folder_name):
    for idx, (line, label) in enumerate(zip(data, labels)):
        xmin = float(line[header.index('XMIN')])
        ymin = float(line[header.index('YMIN')])
        xmax = float(line[header.index('XMAX')])
        ymax = float(line[header.index('YMAX')])
        image_path = line[header.index('original_img_path')]

        # Copy image and rename it
        new_image_name = f'{idx+1}.jpg'
        new_image_path = os.path.join(folder_name, new_image_name)
        shutil.copy(image_path, new_image_path)

        # Create label text file
        label_file_name = f'{idx+1}.txt'
        label_file_path = os.path.join(folder_name.replace("images", "labels"), label_file_name)
        with open(label_file_path, 'w') as label_file:
            label_file.write(f"{label} {xmin} {ymin} {xmax} {ymax}")

# Process train data
train_image_folder = 'images/train'
train_label_folder = 'labels/train'
process_data(train_data, train_labels, train_image_folder)
print("Train dataset organization complete!")

# Process validation data
val_image_folder = 'images/val'
val_label_folder = 'labels/val'
process_data(val_data, val_labels, val_image_folder)
print("Validation dataset organization complete!")
