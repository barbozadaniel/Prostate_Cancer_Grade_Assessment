from datetime import datetime
from logging import Logger
import os
from typing import Dict
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from models.resnet_models import ResNetModel
from data_module import PandaDataModule
from lightning_module import LightningModel
from hyperparameters import HyperParameters
import lightning as L
import lightning.pytorch as LP
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Initialization
SEED: int = 123
NUM_WORKERS: int = 8

TRAINING_DATA_FOLDER: str = 'train-nontiled-prostate-1x512x512'
TESTING_DATA_FOLDER: str = 'test-nontiled-prostate-1x512x512'
DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')
TRAINED_MODELS_FOLDER: str = os.path.abspath('./trained_models')
OUTPUT_LOG_FOLDER: str = './logs'
TRAIN_DATA_CSV_PATH: str = os.path.join(DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER, 'train.csv')

assert TRAINING_DATA_FOLDER.split('-')[-1].split('x')[0] == TESTING_DATA_FOLDER.split('-')[-1].split('x')[0], \
    "Mismatch in Number of tiles (N) from the training and testing sets selected for this run."

assert TRAINING_DATA_FOLDER.split('-')[-1].split('x')[-1] == TESTING_DATA_FOLDER.split('-')[-1].split('x')[-1], \
    "Mismatch in the Tile size from the training and testing sets selected for this run."

NUM_TILES: int = int(TRAINING_DATA_FOLDER.split('-')[-1].split('x')[0])
TILE_SIZE: int = int(TRAINING_DATA_FOLDER.split('-')[-1].split('x')[-1])

def main():
    L.seed_everything(SEED)
    h_params: HyperParameters = HyperParameters(backbone='resnet50',
                                                head='basic',
                                                batch_size=6,
                                                learning_rate=1e-3,
                                                num_tiles=NUM_TILES,
                                                tile_size=TILE_SIZE,
                                                c_out=6,
                                                num_epochs=30)

    model = ResNetModel(c_out=h_params.c_out,
                        n_tiles=h_params.num_tiles,
                        tile_size=h_params.tile_size,
                        backbone=h_params.backbone,
                        head=h_params.head)

    experiment_name: str = f'{h_params.backbone}'

    k_fold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
    df_train_data = df_train_data[df_train_data.is_present == 1]

    df_train_data = df_train_data.sample(frac = 10 / 100)

    cv_splits = k_fold.split(df_train_data, df_train_data['isup_grade'])
    date = datetime.now().strftime('%Y%m%d-%H%M%S')

    lightning_model: LightningModel = LightningModel(model=model,
                                                     h_params=h_params.__dict__)

    for fold, (list_train_idx, list_val_idx) in enumerate(cv_splits):
        print(f'Fold {fold + 1}')

        # Defining clearly to the tensorboard logger in order to put every fold under the same directory.
        tb_logger: Logger = TensorBoardLogger(save_dir=OUTPUT_LOG_FOLDER,
                                              name=f'{experiment_name}-{date}',
                                              version=f'fold_{fold + 1}')

        # Define what metric the checkpoint should track (can be anything returned from the validation_end method)
        checkpoint_callback: LP.Callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                                           filename="/{epoch:02d}-{kappa:.4f}",
                                                           monitor='kappa', mode='max')

        data_module: L.LightningDataModule = PandaDataModule(train_folder_path=os.path.join(DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER),
                                                             test_folder_path=os.path.join(DATASET_FOLDER_PATH, TESTING_DATA_FOLDER),
                                                             list_train_idx=list_train_idx,
                                                             list_val_idx=list_val_idx,
                                                             num_workers=NUM_WORKERS,
                                                             h_params=h_params)

        trainer: L.Trainer = L.Trainer(max_epochs=h_params.num_epochs,
                                       accelerator='gpu',
                                       devices=1,
                                       logger=tb_logger,
                                    #    deterministic=True,
                                       accumulate_grad_batches=1,
                                       callbacks=[checkpoint_callback])

        trainer.fit(model=lightning_model, datamodule=data_module)

        # Creating the folder to store the trained model
        final_model_folder: str = os.path.join(TRAINED_MODELS_FOLDER, f'{experiment_name}-{date}')
        os.makedirs(final_model_folder, exist_ok=True)

        # Saving the final model weights for this training fold
        torch.save(model.state_dict(), os.path.join(TRAINED_MODELS_FOLDER, f'{experiment_name}-{date}/fold_{fold}.pth'))

        # break


if __name__ == '__main__':
    main()
