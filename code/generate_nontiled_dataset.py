# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\\Github\\Prostate_Cancer_Grade_Assessment\\openslide-bin-4.0.0.2-windows-x64\\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide
import multiprocessing
import os
import random
from multiprocessing import Pool
from typing import Dict, List
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import skimage
import tqdm
from tqdm.contrib.concurrent import process_map

from visual_helpers import create_tiles

IS_DEBUG = True

DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')
PANDA_DATASET_NAME: str = 'prostate-cancer-grade-assessment'
PANDA_DATASET_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, PANDA_DATASET_NAME)
PANDA_IMAGE_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_images')
PANDA_MASKS_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_label_masks')
TRAIN_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train.csv')
TEST_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'test.csv')

df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
list_image_ids: List[str] = list(df_train_data[df_train_data.is_present == 1].image_id.values)

NUM_TILES = 1
TILE_SIZE = 512
LEVEL_DIMS = 1

NON_TILED_DATASET_NAME = f'nontiled-prostate-{NUM_TILES}x{TILE_SIZE}x{TILE_SIZE}'
NON_TILED_DATASET_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, NON_TILED_DATASET_NAME)
NON_TILED_IMAGE_FOLDER_PATH = os.path.join(NON_TILED_DATASET_FOLDER_PATH, 'images')
NON_TILED_MASKS_FOLDER_PATH = os.path.join(NON_TILED_DATASET_FOLDER_PATH, 'masks')

# Creating the output folder(s) for the new tiles dataset
os.makedirs(NON_TILED_DATASET_FOLDER_PATH, exist_ok=True)
os.makedirs(NON_TILED_IMAGE_FOLDER_PATH, exist_ok=True)
os.makedirs(NON_TILED_MASKS_FOLDER_PATH, exist_ok=True)

def create_resized_clipped_image(image_id, num_tiles, new_size, level_dim=1):
    og_img: np.ndarray = skimage.io.ImageCollection(os.path.join(PANDA_IMAGE_FOLDER_PATH, f'{image_id}.tiff'))[level_dim]
    og_mask: np.ndarray = skimage.io.ImageCollection(os.path.join(PANDA_MASKS_FOLDER_PATH, f'{image_id}_mask.tiff'))[level_dim]

    # Isolating the Red channel of the mask (Only 1st channel viz. Red channel has mask's target values)
    og_mask_r = og_mask[:, :, 0]

    # Setting all values to '1' to create a blob(s) of '1'
    og_mask_r[og_mask_r > 0] = 1

    try:
        # Finding the top and bottom y-locations where the blob is located in the image
        all_1_y_idxs, _ = np.where(og_mask_r == 1)
        
        if all_1_y_idxs.shape[0] > 0:
            top_most_1_y = all_1_y_idxs[0]
            bottom_most_1_y = all_1_y_idxs[-1]

            # Clipping the image from the top and bottom
            og_mask_tb_clipped = og_mask_r[top_most_1_y:bottom_most_1_y, :]

            # Finding the left and right x-locations where the blob is located in the image
            all_1_x_idxs, _ = np.where(og_mask_tb_clipped.T == 1)
            
            if all_1_x_idxs.shape[0] > 0:
                right_most_1_x = all_1_x_idxs[0]
                left_most_1_x = all_1_x_idxs[-1]

                # Creating the clipped images and mask using the top/bottom/left/right bounds of the blob
                og_img_all_clipped = og_img[top_most_1_y:bottom_most_1_y, right_most_1_x:left_most_1_x]
                og_mask_all_clipped = og_mask[top_most_1_y:bottom_most_1_y, right_most_1_x:left_most_1_x, 0]
            else:
                og_img_all_clipped = og_img[top_most_1_y:bottom_most_1_y, :new_size[1]]
                og_mask_all_clipped = og_mask[top_most_1_y:bottom_most_1_y, :new_size[1], 0]
        else:
            og_img_all_clipped = og_img.copy()
            og_mask_all_clipped = og_mask.copy()

        resized_clipped_img = Image.fromarray(og_img_all_clipped).resize(new_size)
        resized_clipped_mask = Image.fromarray(og_mask_all_clipped).resize(new_size)

        cv_img = cv2.cvtColor(np.array(resized_clipped_img), cv2.COLOR_RGB2BGR)
        cv_mask = np.array(resized_clipped_mask)

        resized_file_name: str = f'{image_id}.png'
        cv2.imwrite(os.path.join(NON_TILED_IMAGE_FOLDER_PATH, resized_file_name), cv_img)
        cv2.imwrite(os.path.join(NON_TILED_MASKS_FOLDER_PATH, resized_file_name), cv_mask)
    except Exception as ex:
        print(image_id)
        print(ex)

    return image_id


def run_pool_function(image_id):
    id = create_resized_clipped_image(image_id, num_tiles=NUM_TILES,
                                      new_size=(TILE_SIZE, TILE_SIZE),
                                      level_dim=LEVEL_DIMS)
    return id


if __name__ == '__main__':
    if IS_DEBUG:
        create_resized_clipped_image('aaa5732cd49bffddf0d2b7d36fbb0a83', num_tiles=NUM_TILES,
                                     new_size=(TILE_SIZE, TILE_SIZE),
                                     level_dim=LEVEL_DIMS)
    else:
        with Pool(processes=8) as pool:
            r = process_map(run_pool_function, list_image_ids, max_workers=8, chunksize=4)
