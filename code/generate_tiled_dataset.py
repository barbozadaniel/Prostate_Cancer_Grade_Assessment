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

import cv2
import numpy as np
import pandas as pd
import skimage
import tqdm
from tqdm.contrib.concurrent import process_map
from visual_helpers import create_tiles, get_clipping_bounds

IS_DEBUG = False

DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')
PANDA_DATASET_NAME: str = 'prostate-cancer-grade-assessment'
PANDA_DATASET_FOLDER_PATH: str = os.path.join(DATASET_FOLDER_PATH, PANDA_DATASET_NAME)
PANDA_IMAGE_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_images')
PANDA_MASKS_FOLDER_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train_label_masks')
TRAIN_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'train.csv')
TEST_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'test.csv')
SUSPICIOUS_DATA_CSV_PATH: str = os.path.join(PANDA_DATASET_FOLDER_PATH, 'suspicious_data.csv')

NUM_TILES = 36
TILE_SIZE = 256
LEVEL_DIMS = 1      # Medium resolution

TILED_DATASET_NAME = f'closetiled-prostate-{NUM_TILES}x{TILE_SIZE}x{TILE_SIZE}'
TILED_DATASET_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, TILED_DATASET_NAME)
TILED_IMAGE_FOLDER_PATH = os.path.join(TILED_DATASET_FOLDER_PATH, 'images')
TILED_MASKS_FOLDER_PATH = os.path.join(TILED_DATASET_FOLDER_PATH, 'masks')

# Creating the output folder(s) for the new tiles dataset
os.makedirs(TILED_DATASET_FOLDER_PATH, exist_ok=True)
os.makedirs(TILED_IMAGE_FOLDER_PATH, exist_ok=True)
os.makedirs(TILED_MASKS_FOLDER_PATH, exist_ok=True)

# Loading the CSV files
df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
df_suspicious_data: pd.DataFrame = pd.read_csv(SUSPICIOUS_DATA_CSV_PATH)

# Excluding data rows corresponding to suspicious/erroneous images
df_train_data = df_train_data.loc[~df_train_data.image_id.isin(df_suspicious_data.image_id.values)]
df_train_data = df_train_data.reset_index(drop=True).copy()

# Exporting the Train DataFrame to a CSV
df_train_data.to_csv(os.path.join(TILED_DATASET_FOLDER_PATH,
                                  os.path.basename(TRAIN_DATA_CSV_PATH)),
                     index=False)

# Creating a list of Image IDs
list_image_ids: List[str] = list(df_train_data.image_id.values)

def create_image_tiles(image_id, num_tiles, tile_size, level_dim=1, is_clipping=False):
    # Reading the TIFF image and mask files
    og_img: np.ndarray = skimage.io.ImageCollection(os.path.join(PANDA_IMAGE_FOLDER_PATH, f'{image_id}.tiff'))[level_dim]
    og_mask: np.ndarray = skimage.io.ImageCollection(os.path.join(PANDA_MASKS_FOLDER_PATH, f'{image_id}_mask.tiff'))[level_dim]

    clipped_og_img: np.ndarray = og_img.copy()
    clipped_og_mask: np.ndarray = og_mask.copy()

    # Clipping images to reduce white space (before tiling)
    if is_clipping:
        x_b, y_b = get_clipping_bounds(og_img)
        clipped_og_img = og_img[x_b[0]:x_b[1], y_b[0]:y_b[1]]
        clipped_og_mask = og_mask[x_b[0]:x_b[1], y_b[0]:y_b[1]]

    # Creating Tiles of required size from a larger image
    list_tiles: List[Dict[any]] = create_tiles(clipped_og_img, clipped_og_mask, num_tiles, tile_size)

    # Saving tile file images & masks to the new Tiled Dataset folder
    for t in list_tiles:
        img_tile, mask_tile, idx = t['img'], t['mask'], t['idx']
        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
        mask_tile = mask_tile[:,:,0]

        tile_file_name: str = f'{image_id}_{idx}.png'
        cv2.imwrite(os.path.join(TILED_IMAGE_FOLDER_PATH, tile_file_name), img_tile)
        cv2.imwrite(os.path.join(TILED_MASKS_FOLDER_PATH, tile_file_name), mask_tile)
    return image_id


def run_pool_function(image_id):
    id = create_image_tiles(image_id, num_tiles=NUM_TILES,
                              tile_size=TILE_SIZE, level_dim=LEVEL_DIMS,
                              is_clipping=True)
    return id


if __name__ == '__main__':
    if IS_DEBUG:
        create_image_tiles(list_image_ids[0], num_tiles=NUM_TILES,
                           tile_size=TILE_SIZE, level_dim=LEVEL_DIMS,
                           is_clipping=True)
    else:
        with Pool(processes=8) as pool:
            r = process_map(run_pool_function, list_image_ids, max_workers=8, chunksize=4)
