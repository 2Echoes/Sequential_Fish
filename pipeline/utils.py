import os
import pandas as pd
import numpy as np
from ..tools import open_image

def open_location(
        Acquisition : pd.DataFrame,
        location : str,
) :
    """
    Open all cycles of a location and reorder stacks in order (cycle,z,y,x,channel)
    """
    loc_Acquisition = Acquisition.loc[Acquisition['location'] == location].index
    assert len(loc_Acquisition) == 1, "Duplicates locations or no location found"

    fish_path = Acquisition.at[loc_Acquisition[0], 'full_path']
    fish_path_list = os.listdir(fish_path)
    fish_path_list.sort() # THIS MUST GIVE CYCLE ORDERED LIST ie : filename cycle matches map cycles and rest of filename doesn't change list order.
    fish_im = open_image(fish_path + fish_path_list[0]) #Opening first tiff file will open all tiff files of this location (multitif_file) with correct reshaping. Ignoring first dim which will be the cycles gives us image dimension

    raise NotImplementedError("#TODO : ADD REORDER STACK")    #TODO : ADD REORDER STACK

    return fish_im

def open_cycle(
        Acquisition : pd.DataFrame,
        location : str,
        cycle : int,
) :
    """
    Open specific cycle of a location and reorder stacks in order (z,y,x,channel)
    """
    loc_Acquisition = Acquisition.loc[Acquisition['location'] == location].index
    assert len(loc_Acquisition) == 1, "Duplicates locations or no location found"
    fish_path = Acquisition.at[loc_Acquisition[0], 'full_path']

    fish_path_list = os.listdir(fish_path)
    fish_path_list.sort() # THIS MUST GIVE CYCLE ORDERED LIST ie : filename cycle matches map cycles and rest of filename doesn't change list order.


    raise NotImplementedError("#TODO : ADD REORDER STACK")    #TODO : ADD REORDER STACK
