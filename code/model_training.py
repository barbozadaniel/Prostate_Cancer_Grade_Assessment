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

TRAINING_DATA_FOLDER: str = 'train-tiled-prostate-36x256x256'
TESTING_DATA_FOLDER: str = 'test-tiled-prostate-36x256x256'
DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath(''), 'dataset')
TRAINED_MODELS_FOLDER: str = './trained_models'
OUTPUT_LOG_FOLDER: str = './logs'
TRAIN_DATA_CSV_PATH: str = os.path.join(
    DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER, 'train.csv')


def main():
    L.seed_everything(SEED)
    h_params: HyperParameters = HyperParameters(backbone='resnext50_semi',
                                                head='basic',
                                                batch_size=2,
                                                learning_rate=1e-4,
                                                num_tiles=36,
                                                tile_size=256,
                                                c_out=6,
                                                num_epochs=2)

    model = ResNetModel(c_out=h_params.c_out,
                        n_tiles=h_params.num_tiles,
                        tile_size=h_params.tile_size,
                        backbone=h_params.backbone,
                        head=h_params.head)

    experiment_name: str = f'{h_params.backbone}'

    k_fold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
    cv_splits = k_fold.split(df_train_data[df_train_data.is_present == 1],
                             df_train_data[df_train_data.is_present == 1]['isup_grade'])
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
        checkpoint_callback: LP.Callback = ModelCheckpoint(dirpath=tb_logger.log_dir + "/{epoch:02d}-{kappa:.4f}",
                                                           monitor='kappa', mode='max')

        data_module: L.LightningDataModule = PandaDataModule(train_folder_path=os.path.join(DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER),
                                                             test_folder_path=os.path.join(DATASET_FOLDER_PATH, TESTING_DATA_FOLDER),
                                                             list_train_idx=list_train_idx,
                                                             list_val_idx=list_val_idx,
                                                             num_workers=NUM_WORKERS,
                                                             h_params=h_params)

        trainer: L.Trainer = L.Trainer(max_epochs=50,
                                       accelerator='cpu',
                                       devices=1,
                                       logger=tb_logger,
                                       deterministic=True,
                                       accumulate_grad_batches=1,
                                       callbacks=[checkpoint_callback])

        trainer.fit(model=lightning_model, datamodule=data_module)

        torch.save(model.model.state_dict(), f'{TRAINED_MODELS_FOLDER}/{experiment_name}-{date}/fold_{fold}.pth')

        break


if __name__ == '__main__':
    main()
