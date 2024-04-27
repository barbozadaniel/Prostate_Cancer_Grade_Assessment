import os
from typing import List
import numpy as np
import pandas as pd
import shutil
import glob

# Initializing file paths and parameters
SRC_DATASET_NAME: str = 'closetiled-prostate-36x256x256'
DEST_TRAIN_DATASET_NAME: str = f'train-{SRC_DATASET_NAME}'
DEST_TEST_DATASET_NAME: str = f'test-{SRC_DATASET_NAME}'
NUM_TILES, TILE_SIZE = SRC_DATASET_NAME.split('-')[-1].split('x')[:2]
NUM_TILES = int(NUM_TILES)
TILE_SIZE = int(TILE_SIZE)
IS_TILED: bool = False if NUM_TILES == 1 else True

DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')

SRC_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, SRC_DATASET_NAME, 'images')
SRC_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, SRC_DATASET_NAME, 'masks')
SRC_TRAIN_DATA_CSV_PATH: str = os.path.join(DATASET_FOLDER_PATH, SRC_DATASET_NAME, 'train.csv')

DEST_TRAIN_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME, 'images')
DEST_TRAIN_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME, 'masks')
DEST_TEST_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME, 'images')
DEST_TEST_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME, 'masks')

PANDA_DATASET_NAME: str = 'prostate-cancer-grade-assessment'
PANDA_DATASET_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, PANDA_DATASET_NAME)
PANDA_IMAGE_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_images')
PANDA_MASKS_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_label_masks')
# TRAIN_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train.csv')
# TEST_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'test.csv')
SUSPICIOUS_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'suspicious_data.csv')

PERCENT_TEST_SET: int = 10

# Loading the CSV files
df_train_data: pd.DataFrame = pd.read_csv(SRC_TRAIN_DATA_CSV_PATH)
df_suspicious_data: pd.DataFrame = pd.read_csv(SUSPICIOUS_DATA_CSV_PATH)

# Excluding data rows corresponding to suspicious/erroneous images
df_train_data = df_train_data.loc[~df_train_data.image_id.isin(df_suspicious_data.image_id.values)]
df_train_data = df_train_data.reset_index(drop=True).copy()

list_image_ids: List[str] = list(df_train_data.image_id.values)

# Randomly sampling rows from the DataFrame to create the Hold-out Test sets
print(f'Randomly splitting {PERCENT_TEST_SET}% data for the Hold-out test set ..')
df_test_data: pd.DataFrame = pd.DataFrame([])
df_test_data = df_train_data.sample(frac = PERCENT_TEST_SET / 100).copy()

print(f'Total images in Full Dataset: {len(df_train_data)}')
print(f'Total images in Hold-out Test Dataset: {len(df_test_data)}')

# Dropping the Hold-out rows from the Training DataFrame
df_train_data.drop(index=list(df_test_data.index), inplace=True)
print(f'Total images in Training Dataset: {len(df_train_data)}')

# Creating the folder structure for the Destination datasets (Train and Test splits)
os.makedirs(os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME), exist_ok=True)
os.makedirs(DEST_TRAIN_IMAGES_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TRAIN_MASKS_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TEST_IMAGES_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TEST_MASKS_FOLDER_PATH, exist_ok=True)

# Exporting the Train & Test DataFrames to a CSV
df_train_data.to_csv(os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME,
                                  os.path.basename(SRC_TRAIN_DATA_CSV_PATH)), index=False)
df_test_data.to_csv(os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME,
                                 'test.csv'), index=False)
print('New `train.csv` and `test.csv` files exported successfully! \n')

# Copying the complete source dataset to the destination Training dataset
print('Copying the complete source dataset into the destination Training dataset folder ..')
shutil.copytree(SRC_IMAGES_FOLDER_PATH, DEST_TRAIN_IMAGES_FOLDER_PATH, dirs_exist_ok=True)
shutil.copytree(SRC_MASKS_FOLDER_PATH, DEST_TRAIN_MASKS_FOLDER_PATH, dirs_exist_ok=True)
print('Successfully copied all training data! \n')

print('Moving testing images/masks from the training folder to the testing folder ..')
# Iterating through the Test DataFrame to move the image/mask pairs corresponding to the Test dataset
for i, row in df_test_data.iterrows():
    image_id: str = row.image_id

    if IS_TILED:
        list_tile_files: List[str] = [f'{image_id}_{i}.png' for i in range(0, NUM_TILES)]
        for tile_file in list_tile_files:
            shutil.move(os.path.join(DEST_TRAIN_IMAGES_FOLDER_PATH, tile_file),
                        os.path.join(DEST_TEST_IMAGES_FOLDER_PATH, tile_file))
            shutil.move(os.path.join(DEST_TRAIN_MASKS_FOLDER_PATH, tile_file),
                        os.path.join(DEST_TEST_MASKS_FOLDER_PATH, tile_file))
    else:
        img_file_name: str = f'{image_id}.png'
        shutil.move(os.path.join(DEST_TRAIN_IMAGES_FOLDER_PATH, img_file_name),
                    os.path.join(DEST_TEST_IMAGES_FOLDER_PATH, img_file_name))
        shutil.move(os.path.join(DEST_TRAIN_MASKS_FOLDER_PATH, img_file_name),
                    os.path.join(DEST_TEST_MASKS_FOLDER_PATH, img_file_name))
        
print('Successfully copied all testing data! \n')
