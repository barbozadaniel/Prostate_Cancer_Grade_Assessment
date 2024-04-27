import argparse
from typing import Dict

class HyperParameters():
    def __init__(self, backbone: str, head: str, batch_size: int, learning_rate: float,
                 num_tiles: int, num_tiles_select: int, is_big_image_tile: bool,
                 tile_size: int, c_out: int, num_epochs: int, warmup_factor: int, num_warmup_epochs: int) -> None:
        self.backbone = backbone
        self.head = head
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_tiles = num_tiles
        self.num_tiles_select = num_tiles_select
        self.is_big_image_tile = is_big_image_tile
        self.tile_size = tile_size
        self.c_out = c_out
        self.num_epochs = num_epochs
        self.warmup_factor = warmup_factor
        self.num_warmup_epochs = num_warmup_epochs

    @staticmethod
    def load_from_dict(dict_h: Dict[str, any]):
        backbone = dict_h['backbone']
        head = dict_h['head']
        batch_size = dict_h['batch_size']
        learning_rate = dict_h['learning_rate']
        num_tiles = dict_h['num_tiles']
        num_tiles_select = dict_h['num_tiles_select']
        is_big_image_tile = dict_h['is_big_image_tile']
        tile_size = dict_h['tile_size']
        c_out = dict_h['c_out']
        num_epochs = dict_h['num_epochs']
        warmup_factor = dict_h['warmup_factor']
        num_warmup_epochs = dict_h['num_warmup_epochs']

        hp = HyperParameters(backbone, head, batch_size, learning_rate,
                             num_tiles, num_tiles_select, is_big_image_tile,
                             tile_size, c_out, num_epochs, warmup_factor, num_warmup_epochs)

        return hp

def to_args(d: Dict[str, any]):
    args = argparse.Namespace()

    def to_args_recursive(args, d, prefix=''):
        for k, v in d.items():
            if type(v) == dict:
                to_args_recursive(args, v, prefix=k)
            elif type(v) in [tuple, list]:
                continue
            else:
                if prefix:
                    args.__setattr__(prefix + '_' + k, v)
                else:
                    args.__setattr__(k, v)

    to_args_recursive(args, d)
    return args
