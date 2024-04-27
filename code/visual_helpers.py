import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
from openslide import OpenSlide


def create_big_image_from_tiles(list_tiles: List[Dict[str, any]],
                                num_tiles: int, tile_size: int, key='img') -> np.ndarray:
    n_row_tiles = int(np.sqrt(num_tiles))
    n_dims = 3 if key == 'img' else 1

    big_image: np.ndarray = np.array([])
    if n_dims > 1:
        big_image = np.zeros((tile_size * n_row_tiles, tile_size * n_row_tiles, n_dims))
    else:
        big_image = np.zeros((tile_size * n_row_tiles, tile_size * n_row_tiles))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            idxes = list(range(num_tiles))

            if len(list_tiles) > idxes[i]:
                this_img = list_tiles[idxes[i]][key]
            else:
                if n_dims > 1:
                    this_img = np.ones((tile_size, tile_size, n_dims)).astype(np.uint8) * 255
                else:
                    this_img = np.ones((tile_size, tile_size)).astype(np.uint8) * 255

            h1 = h * tile_size
            w1 = w * tile_size
            big_image[h1:h1+tile_size, w1:w1+tile_size] = this_img

    return big_image


def plot_img_mask_details(slide_img: OpenSlide, slide_mask: OpenSlide,
                          data_source: str = 'radboud', alpha: float = 0.4,
                          show_thumbnail: bool = True, max_size=(400, 400)) -> None:
    if data_source not in ['radboud', 'karolinska']:
        raise Exception(
            "Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide_mask.read_region((0, 0), slide_mask.level_count - 1, slide_mask.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if data_source == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0,
                                               1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif data_source == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        # display(mask_data)

        img_data = slide_img.read_region((0, 0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
        img_data = img_data.convert(mode='RGB')
        img_data.thumbnail(size=max_size, resample=0)
        # display(img_data)

        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if data_source == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif data_source == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

        alpha_content = Image.fromarray(alpha_content)

        overlayed_image = Image.composite(image1=img_data, image2=mask_data, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        display(Image.fromarray(np.hstack((img_data, mask_data, overlayed_image))))

    # Compute microns per pixel (openslide gives resolution in centimeters)
    spacing = 1 / (float(slide_mask.properties['tiff.XResolution']) / 10000)

    print(f"File id: {slide_mask}")
    print(f"Dimensions: {slide_mask.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide_mask.level_count}")
    print(f"Downsample factor per level: {slide_mask.level_downsamples}")
    print(f"Dimensions of levels: {slide_mask.level_dimensions}")


def create_tiles(img: np.ndarray, mask: np.ndarray,
                 num_tiles: int, tile_size: int) -> List[Dict[str, any]]:
    shape = img.shape
    pad0, pad1 = (tile_size - shape[0] % tile_size) % tile_size, \
                 (tile_size - shape[1] % tile_size) % tile_size

    # Padding the input image/mask to make it divisible by the size of the tile
    img = np.pad(img, [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]],
                 constant_values=255)
    mask = np.pad(mask, [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]],
                  constant_values=0)

    img = img.reshape(img.shape[0]//tile_size, tile_size,
                      img.shape[1]//tile_size, tile_size, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    mask = mask.reshape(mask.shape[0]//tile_size, tile_size,
                        mask.shape[1]//tile_size, tile_size, 3)
    mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    if len(img) < num_tiles:
        mask = np.pad(mask, [[0, num_tiles-len(img)],
                      [0, 0], [0, 0], [0, 0]], constant_values=0)
        img = np.pad(img, [[0, num_tiles-len(img)], [0, 0],
                     [0, 0], [0, 0]], constant_values=255)

    # Creating the output list of tiles object
    list_tiles: List[Dict[str, any]] = []
    
    # Choosing the tiles with the highest tissue content (sum total of pixels in ascending order)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles]
    img = img[idxs]
    mask = mask[idxs]
    for i, _ in enumerate(idxs):
        list_tiles.append({'img': img[i], 'mask': mask[i], 'idx': i})

    return list_tiles


def create_tiles_object_from_images(folder_path: str,
                                    list_tile_img_files: List[str],
                                    list_tile_mask_files: List[str] = [],
                                    include_mask: bool = False,
                                    shuffle: bool = False,
                                    remove_bad_images: bool = False) -> List[Dict[str, any]]:
    list_tiles: List[Dict[str, any]] = []

    if include_mask:
        for i, (tile_img_file, tile_mask_file) in enumerate(zip(list_tile_img_files, list_tile_mask_files)):
            tile_img = np.asarray(Image.open(os.path.join(folder_path, 'images', tile_img_file)))
            tile_mask = np.asarray(Image.open(os.path.join(folder_path, 'masks', tile_mask_file)))
            list_tiles.append({'img': tile_img, 'mask': tile_mask, 'idx': i})
    else:
        for i, tile_img_file in enumerate(list_tile_img_files):
            tile_img = np.asarray(Image.open(os.path.join(folder_path, tile_img_file)))
            list_tiles.append({'img': tile_img, 'idx': i})

    return list_tiles


def get_clipping_bounds(og_np_img: np.ndarray, perform_thresholding: bool = False):
    tmp_og_img = og_np_img.copy()

    if not perform_thresholding:
        # Setting background (white) to '0'
        tmp_og_img[tmp_og_img == 255] = 0
    else:
        # Converting RGB image to Grayscale image
        tmp_gs_img = cv2.cvtColor(tmp_og_img, cv2.COLOR_RGB2GRAY)

        # Applying thresdholding for improved image clipping
        th_val, thresh_img = cv2.threshold(tmp_gs_img, 150, 255, cv2.THRESH_BINARY)
        tmp_og_img[thresh_img == 0] = 1
        tmp_og_img[thresh_img == 255] = 0

    # Setting all non-zero pixels to '1'
    tmp_og_img[tmp_og_img > 0] = 1

    # Converting 3-channel images to 1-channel
    if tmp_og_img.ndim == 3:
        tmp_og_1c_img = np.max(tmp_og_img, axis=2)
    else:
        tmp_og_1c_img = tmp_og_img.copy()

    top_most_1_y = 0
    bottom_most_1_y = 0
    right_most_1_x = 0
    left_most_1_x = 0

    # Finding the top and bottom y-locations where the blob is located in the image
    all_1_y_idxs, _ = np.where(tmp_og_1c_img == 1)

    if all_1_y_idxs.shape[0] > 0:
        top_most_1_y = all_1_y_idxs[0]
        bottom_most_1_y = all_1_y_idxs[-1]

        # Clipping the image from the top and bottom
        tmp_og_1c_img_clipped = tmp_og_1c_img[top_most_1_y:bottom_most_1_y, :].copy(
        )

        # Finding the left and right x-locations where the blob is located in the image
        all_1_x_idxs, _ = np.where(tmp_og_1c_img_clipped.T == 1)

        if all_1_x_idxs.shape[0] > 0:
            right_most_1_x = all_1_x_idxs[0]
            left_most_1_x = all_1_x_idxs[-1]
        else:
            pass
    else:
        pass

    return (top_most_1_y, bottom_most_1_y), (right_most_1_x, left_most_1_x)


def get_image_bounds_coverage(og_np_img: np.ndarray,
                              x_bounds: Tuple[int, int], y_bounds: Tuple[int, int]) -> float:
    coverage: float = 0
    coverage = (abs(x_bounds[1] - x_bounds[0]) * abs(y_bounds[1] -
                y_bounds[0])) / (og_np_img.shape[0] * og_np_img.shape[1])
    return coverage
