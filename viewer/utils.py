import numpy as np
import os
from typing import Literal
from skimage import io
from tqdm import tqdm

def open_image(fullpath, image_number=1) :
    arrays = [io.imread(fullpath, plugin="tifffile", img_num=im_index) for im_index in range(image_number)]



    arrays = np.stack(arrays)
    return arrays

def open_segmentation(segmentation_folder_fullpath: str, locations : 'list[str]', object : Literal['nucleus','cytoplasm'], z_repeat=None) :
    """
    Open with sorting on 'location'
    """

    masks = []

    #Opening masks
    for location in tqdm(locations, desc = "opening {0} masks".format(object)) :
        new_mask = np.load(segmentation_folder_fullpath + '/{0}_segmentation.npz'.format(location))[object]

        if type(z_repeat) != type(None) :
            new_mask = np.repeat(
                new_mask[np.newaxis],
                repeats= z_repeat,
                axis=0
            )
        masks.append(new_mask)

    masks = np.stack(masks)
    return masks

def pad_to_shape(array : np.ndarray, new_shape) :
    shape = array.shape

    if len(array.shape) != len(new_shape) : raise ValueError("dimensions of array and new_shape don't match")

    pad_width_list = []

    for axis, axis_size in enumerate(shape) :
        target_size = new_shape[axis]
        
        pad_width = int(target_size - axis_size)
        if pad_width >= 0 :
            pad_width_list.append([0,pad_width])
        else :
            raise ValueError("Can't pad to new size {0} on axis {1} because current size {2} is bigger.".format(target_size, axis, axis_size))
    
    array = np.pad(array, pad_width_list)

    return array