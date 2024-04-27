import os
from typing import List

import numpy as np
import pandas as pd
import PIL.Image as Image
import lightning as L
import torch
import torchvision.transforms as transforms
import torch.utils.data as D
from hyperparameters import HyperParameters
from random import sample
from visual_helpers import create_tiles_object_from_images

class TileDataset(D.Dataset):
    def __init__(self, dataset_folder_path: str, df_data: pd.DataFrame,
                 num_tiles: int, num_tiles_select: int, tile_size: int,
                 is_big_image_tile: bool = False, transform=None):
        """
        dataset_folder_path: Folder path where the Dataset is located
        df_data: DataFrame containing image metadata
        num_tiles: Number of tiles to be returned by the Dataset
        transform: The function to apply to the image. Usually data augmentation.
        """
        self.dataset_folder_path = dataset_folder_path
        self.df_data = df_data
        self.num_tiles = num_tiles
        self.num_tiles_select = num_tiles_select
        self.tile_size = tile_size
        self.is_big_image_tile = is_big_image_tile
        self.list_images: List[str] = self.df_data['image_id'].values

        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.list_images[idx]
        metadata = self.df_data.iloc[idx]

        if self.num_tiles != 1:
            # Tiled: Single/Multiple images as input (image generated by creating tiles from TIFF images)
            if not self.is_big_image_tile:
                # Multiple images as input (Stack of image tiles)
                list_image_tiles = []
                list_idx = list(range(0, self.num_tiles))
                list_tile_file_names = [f'{image_id}_{str(i)}.png'
                                        for i in sample(list_idx, self.num_tiles_select)]
                for tile_file in list_tile_file_names:
                    image = Image.open(os.path.join(self.dataset_folder_path, tile_file))

                    if self.transform is not None:
                        image = self.transform(image)

                    image = transforms.ToTensor()(1 - np.array(image))
                    list_image_tiles.append(image)

                image_stack = torch.stack(list_image_tiles, dim=0)

                return {'image': image_stack,
                        'data_provider': metadata['data_provider'],
                        'isup_grade': metadata['isup_grade'],
                        'gleason_score': metadata['gleason_score'],
                        'target': metadata['isup_grade']}
            else:
                # Single big image as input (N x N big image of tiles)
                list_idx = list(range(0, self.num_tiles))
                list_tile_file_names = [f'{image_id}_{str(i)}.png' for i in list_idx]
                list_tiles = create_tiles_object_from_images(self.dataset_folder_path, list_tile_file_names,
                                                             include_mask=False,
                                                             shuffle=True,
                                                             remove_bad_images=True)
                
                num_rows_cols: int = int(np.sqrt(self.num_tiles))
                big_image = np.zeros((self.tile_size * num_rows_cols,
                                      self.tile_size * num_rows_cols, 3))
                
                # Creating a big image from the tiles of the original image
                for h in range(num_rows_cols):
                    for w in range(num_rows_cols):
                        i = h * num_rows_cols + w
            
                        if len(list_tiles) > list_idx[i]:
                            tile_image = list_tiles[list_idx[i]]['img']
                        else:
                            tile_image = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                        tile_image = 255 - tile_image

                        if self.transform is not None:
                            tile_image = self.transform(Image.fromarray(tile_image))

                        h1 = h * self.tile_size
                        w1 = w * self.tile_size
                        big_image[h1:h1+self.tile_size, w1:w1+self.tile_size] = tile_image.numpy().T if not isinstance(tile_image, np.ndarray) else tile_image

                # Applying the transformations (for augmentations, if any)
                if self.transform is not None:
                    big_image = self.transform(Image.fromarray((big_image * 255).astype(np.uint8)))
                
                # Normalizing the image
                if isinstance(big_image, np.ndarray):
                    big_image /= 255
                    big_image = big_image.astype(np.float32)
                else:
                    big_image = big_image.numpy().T.astype(np.float32)
                # big_image /= 255
                big_image = big_image.transpose(2, 0, 1)

                # Setting up the target label
                label = np.zeros(5).astype(np.float32)
                label[:metadata['isup_grade']] = 1.

                return {'image': torch.tensor(big_image),
                        'data_provider': metadata['data_provider'],
                        'isup_grade': metadata['isup_grade'],
                        'gleason_score': metadata['gleason_score'],
                        'target': torch.tensor(label)}
        else:
            # Non-Tiled: Single image as input (image generated by resizing original TIFF images)
            image = Image.open(os.path.join(self.dataset_folder_path, f'{image_id}.png'))
            if self.transform is not None:
                image = self.transform(image)
                image = 1 - image
                image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                             [0.1279171, 0.24528177, 0.16098117])(image)
                return {'image': image,
                        'data_provider': metadata['data_provider'],
                        'isup_grade': metadata['isup_grade'],
                        'gleason_score': metadata['gleason_score'],
                        'target': metadata['isup_grade']}

    def __len__(self):
        return len(self.list_images)


class PandaDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_folder_path: str = "./",
        test_folder_path: str = "./",
        list_train_idx: List[int] = [],
        list_val_idx: List[int] = [],
        num_workers: int = 0,
        h_params: HyperParameters = None
    ):
        super().__init__()
        self.train_folder_path = train_folder_path
        self.test_folder_path = test_folder_path
        self.list_train_idx = list_train_idx
        self.list_val_idx = list_val_idx
        self.num_workers = num_workers
        self.h_params = h_params

        self.train_transform = None
        self.test_transform = None

        self.train_dataset: D.Dataset = None
        self.val_dataset: D.Dataset = None
        self.test_dataset: D.Dataset = None

    def prepare_data(self):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        return

    def setup(self, stage=None):
        df_train_data: pd.DataFrame = pd.read_csv(os.path.join(self.train_folder_path, 'train.csv'))
        df_train_data = df_train_data[df_train_data.is_present == 1]

        df_test_data: pd.DataFrame = pd.read_csv(os.path.join(self.test_folder_path, 'test.csv'))
        df_test_data = df_test_data[df_test_data.is_present == 1]

        self.train_dataset = TileDataset(os.path.join(self.train_folder_path, 'images'),
                                         df_train_data.iloc[self.list_train_idx],
                                         self.h_params.num_tiles, self.h_params.num_tiles_select,
                                         self.h_params.tile_size, self.h_params.is_big_image_tile, self.train_transform)

        self.val_dataset = TileDataset(os.path.join(self.train_folder_path, 'images'),
                                       df_train_data.iloc[self.list_val_idx],
                                       self.h_params.num_tiles, self.h_params.num_tiles_select,
                                       self.h_params.tile_size, self.h_params.is_big_image_tile, self.test_transform)

        self.test_dataset = TileDataset(os.path.join(self.test_folder_path, 'images'),
                                        df_test_data,
                                        self.h_params.num_tiles, self.h_params.num_tiles_select,
                                        self.h_params.tile_size, self.h_params.is_big_image_tile, self.test_transform)

    def train_dataloader(self):
        train_loader = D.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = D.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = D.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = D.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        return predict_loader