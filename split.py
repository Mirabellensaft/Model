import os
import shutil
import pandas as pd
from sklearn import model_selection

# Define paths
csv_path = "styles.csv"
image_folder = "archive/e-commerce/images"
train_dir = "train"
val_dir = "val"
test_dir = "test"

# Create directories if they don't exist
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Read CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Identify rare labels (e.g., labels with less than two samples)
rare_labels = df['articleType'].value_counts()[df['articleType'].value_counts() < 10].index.tolist()

# Merge rare labels into a single category or distribute them among existing labels
# For example, you can merge rare labels into a 'Other' category
df.loc[df['articleType'].isin(rare_labels), 'articleType'] = 'Other'

# Calculate class frequencies
class_frequencies = df['articleType'].value_counts(normalize=True)
print(class_frequencies)

# Define initial splitting ratios (adjust as needed)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Adjust splitting ratios for rare labels
threshold = 0.001  # Set a threshold for rare labels
rare_labels = class_frequencies[class_frequencies < threshold].index.tolist()

# Calculate the number of items for rare labels
num_rare_items = sum(df['articleType'].isin(rare_labels))

# Calculate the total number of items for adjustment
total_items = len(df)

# Calculate the adjustment factor
adjustment_factor = num_rare_items / total_items

# Adjust splitting ratios
train_ratio += adjustment_factor * 0.2  # Increase train_ratio
val_ratio -= adjustment_factor * 0.1   # Decrease val_ratio
test_ratio -= adjustment_factor * 0.1  # Decrease test_ratio

# Stratified splitting
train_df, temp_df = model_selection.train_test_split(df, train_size=train_ratio, stratify=df['articleType'])
val_df, test_df = model_selection.train_test_split(temp_df, train_size=val_ratio/(val_ratio + test_ratio), stratify=temp_df['articleType'])


# Check if rare articleTypes are adequately represented in each subset
train_label_frequencies = train_df['articleType'].value_counts(normalize=True)
val_label_frequencies = val_df['articleType'].value_counts(normalize=True)
test_label_frequencies = test_df['articleType'].value_counts(normalize=True)

# Print label frequencies for each subset
print("Training Label Frequencies:")
print(train_label_frequencies)
print("\nValidation Label Frequencies:")
print(val_label_frequencies)
print("\nTesting Label Frequencies:")
print(test_label_frequencies)

# Function to move images to directories based on their labels
def move_images(df, source_folder, destination_folder):
    for index, row in df.iterrows():
        image_id = row['id']
        label = row['articleType']
        
        # Create subdirectory for each class if it doesn't exist
        class_dir = os.path.join(destination_folder, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        # Move image to the appropriate class directory
        src_path = os.path.join(source_folder, f"{image_id}.jpg")  # Assuming images have ".jpg" extension
        dest_path = os.path.join(class_dir, f"{image_id}.jpg")
        shutil.move(src_path, dest_path)

# Function to move images to directories based on their labels
def move_test_images(df, source_folder, destination_folder):
    for index, row in df.iterrows():
        image_id = row['id']
        label = row['articleType']
        
        # # Create subdirectory for each class if it doesn't exist
        # class_dir = os.path.join(destination_folder, str(label))
        # os.makedirs(class_dir, exist_ok=True)
        
        # Move image to the appropriate class directory
        src_path = os.path.join(source_folder, f"{image_id}.jpg")  # Assuming images have ".jpg" extension
        dest_path = os.path.join(destination_folder, f"{image_id}.jpg")
        shutil.move(src_path, dest_path)


# Move images to train directory
move_images(train_df, image_folder, train_dir)

# Move images to validation directory
move_images(val_df, image_folder, val_dir)

# Move images to test directory
move_images(test_df, image_folder, test_dir)
