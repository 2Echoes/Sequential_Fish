import os
import pandas as pd
from ..tools import open_image

def open_fish_signal(
        Acquisition : pd.DataFrame,
        location : str,
) :
    loc_Acquisition = Acquisition.loc[Acquisition['location'] == location].index
    assert len(loc_Acquisition) == 1, "Duplicates locations or no location found"

    fish_path = Acquisition.at[loc_Acquisition[0], 'full_path'] + "" #TODO put folder name for fish signal
    fish_path_list = os.listdir(fish_path)
    fish_path_list.sort() # THIS MUST GIVE CYCLE ORDERED LIST ie : filename cycle matches map cycles and rest of filename doesn't change list order.
    fish_im = open_image(fish_path + fish_path_list[0]) #Opening first tiff file will open all tiff files of this location (multitif_file) with correct reshaping. Ignoring first dim which will be the cycles gives us image dimension

    return fish_im