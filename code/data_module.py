import os
from typing import List

import pandas as pd
import PIL.Image as Image
import lightning as L
import torch
import torchvision.transforms as transforms
import torch.utils.data as D
from hyperparameters import HyperParameters

class TileDataset(D.Dataset):
    def __init__(self, dataset_folder_path: str, df_data: pd.DataFrame, num_tiles: int, transform=None):
        """
        dataset_folder_path: Folder path where the Dataset is located
        df_data: DataFrame containing image metadata
        num_tiles: Number of tiles to be returned by the Dataset
        transform: The function to apply to the image. Usually data augmentation.
        """
        self.dataset_folder_path = dataset_folder_path
        self.df_data = df_data
        self.num_tiles = num_tiles
        self.list_images: List[str] = self.df_data['image_id'].values

        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.list_images[idx]
        metadata = self.df_data.iloc[idx]

        if self.num_tiles != 1:
            image_tiles = []
            tiles = [f'{image_id}_{str(i)}.png' for i in range(0, self.num_tiles)]
            for tile in tiles:
                image = Image.open(os.path.join(self.dataset_folder_path, tile))

                if self.transform is not None:
                    image = self.transform(image)

                image = 1 - image
                image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                             [0.1279171, 0.24528177, 0.16098117])(image)
                image_tiles.append(image)

            image_tiles = torch.stack(image_tiles, dim=0)

            return {'image': image_tiles,
                    'data_provider': metadata['data_provider'],
                    'isup_grade': metadata['isup_grade'],
                    'gleason_score': metadata['gleason_score']}
        else:
            image = Image.open(os.path.join(self.dataset_folder_path, f'{image_id}.png'))
            if self.transform is not None:
                image = self.transform(image)
                image = 1 - image
                image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
                                             [0.1279171, 0.24528177, 0.16098117])(image)
                return {'image': image,
                        'data_provider': metadata['data_provider'],
                        'isup_grade': metadata['isup_grade'],
                        'gleason_score': metadata['gleason_score']}

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
                                         df_train_data.iloc[self.list_train_idx], self.h_params.num_tiles, self.train_transform)

        self.val_dataset = TileDataset(os.path.join(self.train_folder_path, 'images'),
                                       df_train_data.iloc[self.list_val_idx], self.h_params.num_tiles, self.test_transform)

        self.test_dataset = TileDataset(os.path.join(self.test_folder_path, 'images'),
                                        df_test_data, self.h_params.num_tiles, self.test_transform)

    def train_dataloader(self):
        train_loader = D.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = D.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = D.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.h_params.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
