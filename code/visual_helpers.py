from typing import List
import numpy as np
from PIL import Image

def create_image_from_tiles(tiles, num_tiles, tile_size, key='img'):
    n_row_tiles = int(np.sqrt(num_tiles))
    n_dims = 3 if key =='img' else 1
    
    if n_dims > 1:
        images = np.zeros((tile_size * n_row_tiles, tile_size * n_row_tiles, n_dims))
    else:
        images = np.zeros((tile_size * n_row_tiles, tile_size * n_row_tiles))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            idxes = list(range(num_tiles))

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]][key]
            else:
                if n_dims > 1:
                    this_img = np.ones((tile_size, tile_size, n_dims)).astype(np.uint8) * 255
                else:
                    this_img = np.ones((tile_size, tile_size)).astype(np.uint8) * 255
            # this_img = 255 - this_img
            # if self.transform is not None:
            #     this_img = self.transform(image=this_img)['image']
            h1 = h * tile_size
            w1 = w * tile_size
            images[h1:h1+tile_size, w1:w1+tile_size] = this_img

    return images

def plot_img_mask_details(slide_img, slide_mask, center='radboud', alpha=0.4, show_thumbnail=True, max_size=(400,400)):
    """Print some basic information about a slide"""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide_mask.read_region((0,0), slide_mask.level_count - 1, slide_mask.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        # display(mask_data)

        img_data = slide_img.read_region((0,0), slide_img.level_count - 1, slide_img.level_dimensions[-1])
        img_data = img_data.convert(mode='RGB')
        img_data.thumbnail(size=max_size, resample=0)
        # display(img_data)

        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
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

def create_tiles(img, mask, num_tiles, tile_size):
    result = []
    shape = img.shape
    pad0,pad1 = (tile_size - shape[0]%tile_size)%tile_size, (tile_size - shape[1]%tile_size)%tile_size
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//tile_size,tile_size,img.shape[1]//tile_size, tile_size, 3)
    img = img.transpose(0,2,1,3,4).reshape(-1,tile_size,tile_size,3)
    mask = mask.reshape(mask.shape[0]//tile_size, tile_size, mask.shape[1]//tile_size, tile_size, 3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,tile_size,tile_size,3)
    if len(img) < num_tiles:
        mask = np.pad(mask,[[0,num_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,num_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:num_tiles]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result

def create_tiles_object_from_images(list_tile_img_files, list_tile_mask_files):
    tiles = []
    for i, (tile_img_file, tile_mask_file) in enumerate(zip(list_tile_img_files, list_tile_mask_files)):
        tile_img = np.asarray(Image.open(tile_img_file))
        tile_mask = np.asarray(Image.open(tile_mask_file))
        tiles.append({'img': tile_img, 'mask': tile_mask, 'idx': i})
    
    return tiles