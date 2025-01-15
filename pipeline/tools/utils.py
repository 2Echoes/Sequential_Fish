import numpy as np
import datetime as dt
from czifile import imread as _imread
from bigfish.stack import read_image as _read_image


class MappingError(Exception) :
    """
    Raised when user inputs an incorrect image mapping.
    """
    pass

def auto_map_channels(image: np.ndarray, color_number: int, cycle_number: int, bead_channel = True) :
    """
    Assume x and y are the biggest dimension.
    """

    dim = image.ndim
    shape = image.shape
    reducing_list = list(shape)
    map_ = dict()

    try :
        c_idx = shape.index(color_number + bead_channel)
    except ValueError :
        raise MappingError("{0} colors channels are expected from experimental file but no matching axis was found in shape {1}.".format(color_number, shape))
    else :
        map_['c'] = c_idx
        reducing_list[c_idx] = -1

    if dim > 4 : #Else dapi is being mapped and has only one cycle.
        try :
            cycles_idx = shape.index(cycle_number)
        except ValueError :
            raise MappingError("{0} cycles are expected from experimental file but no matching axis was found in shape {1}.".format(cycle_number, shape))
        else :
            map_['cycles'] = cycles_idx
            reducing_list[cycles_idx] = -1

    #Set the biggest dimension to y
    y_val = max(reducing_list)
    y_idx = shape.index(y_val)
    map_['y'] = y_idx

    #2nd biggest set to x
    reducing_list[y_idx] = -1
    x_val = max(reducing_list)
    x_idx = reducing_list.index(x_val)
    reducing_list[y_idx] = y_val

    map_['x'] = x_idx
    reducing_list.remove(y_val)
    reducing_list.remove(x_val)
    reducing_list.remove(color_number + bead_channel)
    reducing_list.remove(cycle_number)

    #Remaning value set to z
    z_val = reducing_list[0]
    z_idx = shape.index(z_val)
    map_['z'] = z_idx

    return map_


def reorder_image_stack(image, map) :
    
    dim = image.ndim
    print(map)
    if dim == 5 :
        new_order = (map['cycles'], map['z'], map['y'], map['x'], map['c'])
    elif dim == 4 :
        new_order = (map['z'], map['y'], map['x'], map['c'])

    image = np.moveaxis(new_order, [0,1,2,3,4])
    return image

def open_image(path:str, map=None) :
    """
    Supports czi, png, jpg, jpeg, tif or tiff extensions.
    """

    SUPPORTED_TYPES = ('.png', '.jpg', '.jpeg','.tif', '.tiff')

    if path.endswith('.czi') :
        im = _imread(path)
    elif path.endswith(SUPPORTED_TYPES) :
        im = _read_image(path)
    else :
        raise ValueError("Unsupported type. Currently supported types are {0}".format(SUPPORTED_TYPES))
    
    if type(map) != type(None) :
        im =reorder_image_stack(im, map)

    return im

def get_datetime():
    return dt.datetime.now().strftime("%Y%m%d %H-%M-%S")