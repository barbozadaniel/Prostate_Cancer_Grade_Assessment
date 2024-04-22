import os
from typing import List
import numpy as np
import pandas as pd
import shutil
import glob

# Initializing file paths and parameters
SRC_DATASET_NAME: str = 'tiled-prostate-36x256x256'
DEST_TRAIN_DATASET_NAME: str = f'train-{SRC_DATASET_NAME}'
DEST_TEST_DATASET_NAME: str = f'test-{SRC_DATASET_NAME}'
NUM_TILES, TILE_SIZE = SRC_DATASET_NAME.split('-')[-1].split('x')[:2]
NUM_TILES = int(NUM_TILES)
TILE_SIZE = int(TILE_SIZE)

DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')

SRC_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, SRC_DATASET_NAME, 'images')
SRC_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, SRC_DATASET_NAME, 'masks')
DEST_TRAIN_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME, 'images')
DEST_TRAIN_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME, 'masks')
DEST_TEST_IMAGES_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME, 'images')
DEST_TEST_MASKS_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME, 'masks')

PANDA_DATASET_NAME: str = 'prostate-cancer-grade-assessment'
PANDA_DATASET_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, PANDA_DATASET_NAME)
PANDA_IMAGE_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_images')
PANDA_MASKS_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_label_masks')
TRAIN_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train.csv')
TEST_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'test.csv')

PERCENT_TEST_SET: int = 10

# Loading the 'train.csv' and 'test.csv' into Pandas DataFrames 
df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
df_test_data: pd.DataFrame = pd.read_csv(TEST_DATA_CSV_PATH)
list_image_ids: List[str] = list(df_train_data[df_train_data.is_present == 1].image_id.values)

# # Randomly sampling rows from the DataFrame to create the Hold-out Test sets
# df_test_data: pd.DataFrame = df_train_data[df_train_data.is_present == 1].sample(frac = PERCENT_TEST_SET / 100)

# # Dropping the Hold-out rows from the Training DataFrame
# df_train_data.drop(index=list(df_test_data.index), inplace=True)

# # Exporting the Train & Test DataFrames to a CSV
# df_train_data.to_csv(TRAIN_DATA_CSV_PATH, index=False)
# df_test_data.to_csv(TEST_DATA_CSV_PATH, index=False)

# Creating the folder structure for the Destination datasets (Train and Test splits)
os.makedirs(os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME), exist_ok=True)
os.makedirs(DEST_TRAIN_IMAGES_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TRAIN_MASKS_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TEST_IMAGES_FOLDER_PATH, exist_ok=True)
os.makedirs(DEST_TEST_MASKS_FOLDER_PATH, exist_ok=True)

# # Copying the complete source dataset to the destination Training dataset
# shutil.copytree(SRC_IMAGES_FOLDER_PATH, DEST_TRAIN_IMAGES_FOLDER_PATH, dirs_exist_ok=True)
# shutil.copytree(SRC_MASKS_FOLDER_PATH, DEST_TRAIN_MASKS_FOLDER_PATH, dirs_exist_ok=True)

# Iterating through the Test DataFrame to move the image/mask pairs corresponding to the Test dataset
for i, row in df_test_data.iterrows():
    image_id: str = row.image_id
    list_tile_files: List[str] = [f'{image_id}_{i}.png' for i in range(0, NUM_TILES)]
    for tile_file in list_tile_files:
        shutil.move(os.path.join(DEST_TRAIN_IMAGES_FOLDER_PATH, tile_file),
                    os.path.join(DEST_TEST_IMAGES_FOLDER_PATH, tile_file))
        shutil.move(os.path.join(DEST_TRAIN_MASKS_FOLDER_PATH, tile_file),
                    os.path.join(DEST_TEST_MASKS_FOLDER_PATH, tile_file))

# Removing missing image/mask pairs from the new datasets (if any)
# TODO

# Copying the Train & Test CSV files to the new folders
shutil.copy(TRAIN_DATA_CSV_PATH,
            os.path.join(DATASET_FOLDER_PATH, DEST_TRAIN_DATASET_NAME, os.path.basename(TRAIN_DATA_CSV_PATH)))
shutil.copy(TEST_DATA_CSV_PATH,
            os.path.join(DATASET_FOLDER_PATH, DEST_TEST_DATASET_NAME, os.path.basename(TEST_DATA_CSV_PATH)))
