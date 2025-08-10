import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = (224, 224)
IMG_DIR = 'urban_issues_dataset'
ENCODED_CSV = 'multi_label_encoded.csv'

# Read encoded labels
df = pd.read_csv(ENCODED_CSV)
sub_categories = [col for col in df.columns if col not in ['image_filename', 'main_category', 'sub_category', 'split']]

# Helper to get image path
def get_image_path(row):
    # Try to find the correct subfolder for the image
    for subfolder in os.listdir(IMG_DIR):
        img_path = os.path.join(IMG_DIR, subfolder, row['split'], 'images', row['image_filename'])
        if os.path.exists(img_path):
            return img_path
    return None

def load_images_and_labels(df):
    images = []
    labels = []
    missing = []
    for _, row in df.iterrows():
        img_path = get_image_path(row)
        if img_path:
            img = load_img(img_path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(row[sub_categories].values.astype(np.float32))
        else:
            missing.append(row['image_filename'])
    if missing:
        print(f"Missing images ({len(missing)}):", missing)
    return np.array(images), np.array(labels)

# Split data
df_train = df[df['split'] == 'train']
df_valid = df[df['split'] == 'valid']
df_test = df[df['split'] == 'test']

X_train, y_train = load_images_and_labels(df_train)
X_valid, y_valid = load_images_and_labels(df_valid)
X_test, y_test = load_images_and_labels(df_test)

print(f'Train samples: {len(X_train)}')
print(f'Valid samples: {len(X_valid)}')
print(f'Test samples: {len(X_test)}')

# Save as numpy arrays for fast loading
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_valid.npy', X_valid)
np.save('y_valid.npy', y_valid)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print('Image preprocessing and data loader complete. Numpy arrays saved.')
