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
import yaml

# Initialization
SEED: int = 123
NUM_WORKERS: int = 4

IS_TRAIN = False
IS_TEST = not(IS_TRAIN)

TRAINING_DATA_FOLDER: str = 'train-tiled-prostate-36x256x256'
TESTING_DATA_FOLDER: str = 'test-tiled-prostate-36x256x256'
DATASET_FOLDER_PATH: str = os.path.join(os.path.abspath('./Prostate_Cancer_Grade_Assessment'), 'dataset')
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
    torch.set_float32_matmul_precision('medium')

    L.seed_everything(SEED)
    h_params: HyperParameters = HyperParameters(backbone='resnet50',
                                                head='basic',
                                                batch_size=4,
                                                learning_rate=1e-4,
                                                num_tiles=NUM_TILES,
                                                num_tiles_select=25,
                                                tile_size=TILE_SIZE,
                                                c_out=6,
                                                num_epochs=30)

    model = ResNetModel(c_out=h_params.c_out,
                        n_tiles=h_params.num_tiles_select,
                        tile_size=h_params.tile_size,
                        backbone=h_params.backbone,
                        head=h_params.head)

    experiment_name: str = f'{h_params.backbone}'

    k_fold = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

    df_train_data: pd.DataFrame = pd.read_csv(TRAIN_DATA_CSV_PATH)
    df_train_data = df_train_data[df_train_data.is_present == 1]

    cv_splits = k_fold.split(df_train_data, df_train_data['isup_grade'])
    date = datetime.now().strftime('%Y%m%d-%H%M%S')

    lightning_model: LightningModel = LightningModel(model=model,
                                                     h_params=h_params.__dict__)

    if IS_TRAIN:
        for fold, (list_train_idx, list_val_idx) in enumerate(cv_splits):
            print(f'Fold {fold + 1}')

            # Defining clearly to the tensorboard logger in order to put every fold under the same directory.
            tb_logger: Logger = TensorBoardLogger(save_dir=OUTPUT_LOG_FOLDER,
                                                name=f'{experiment_name}-{date}',
                                                version=f'fold_{fold + 1}')

            # Define what metric the checkpoint should track (can be anything returned from the validation_end method)
            checkpoint_callback: LP.Callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                                            filename="{epoch:02d}-{kappa:.4f}",
                                                            monitor='kappa', mode='max')

            data_module: L.LightningDataModule = PandaDataModule(train_folder_path=os.path.join(DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER),
                                                                test_folder_path=os.path.join(DATASET_FOLDER_PATH, TESTING_DATA_FOLDER),
                                                                list_train_idx=list_train_idx,
                                                                list_val_idx=list_val_idx,
                                                                num_workers=NUM_WORKERS,
                                                                h_params=h_params)

            trainer: L.Trainer = L.Trainer(max_epochs=h_params.num_epochs,
                                        accelerator='gpu',
                                        devices='-1',
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

    elif IS_TEST:
        # t_model_1 = model.load_state_dict('trained_models/resnet50-20240423-000603/fold_1.pth')
        
        chkpt_file_path: str = 'logs/resnet50-20240423-000603/fold_2/epoch=26-kappa=0.7400.ckpt'
        hparams_file_path: str = 'logs/resnet50-20240423-000603/fold_2/hparams.yaml'
        
        # Loading hparams.yaml
        loaded_h_params: Dict[str, any] = {}
        with open(hparams_file_path, 'r') as stream:
            dict_yaml = yaml.safe_load(stream)
            loaded_h_params = dict_yaml['h_params']
        
        # Testing Code
        lt_model = LightningModel.load_from_checkpoint(model=model,
                                    h_params=loaded_h_params,
                                     checkpoint_path=chkpt_file_path,
                                     hparams_file=hparams_file_path,
                                     map_location=None)
        
        data_module: L.LightningDataModule = PandaDataModule(train_folder_path=os.path.join(DATASET_FOLDER_PATH, TRAINING_DATA_FOLDER),
                                                                test_folder_path=os.path.join(DATASET_FOLDER_PATH, TESTING_DATA_FOLDER),
                                                                # list_train_idx=list_train_idx,
                                                                # list_val_idx=list_val_idx,
                                                                num_workers=NUM_WORKERS,
                                                                h_params=h_params)

        trainer: L.Trainer = L.Trainer(max_epochs=h_params.num_epochs,
                                       accelerator='gpu',
                                        devices=1,
                                        num_nodes=1,
                                        # logger=tb_logger,
                                        accumulate_grad_batches=1)
        testing_results = trainer.predict(model=lt_model, datamodule=data_module)
        print(testing_results)


if __name__ == '__main__':
    main()
