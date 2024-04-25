import argparse

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

    def to_args(d):
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