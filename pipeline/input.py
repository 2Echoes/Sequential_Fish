"""
This script aims at reading the input folder and preparing data folders and locations for next scripts.
"""

from Sequential_Fish.pipeline_parameters import RUN_PATH, FOLDER_KEYS, MAP_FILENAME, cycle_regex, CYCLE_KEY, GENES_NAMES_KEY, WASHOUT_KEY_WORD, HAS_BEAD_CHANNEL
from Sequential_Fish.pipeline.tools.utils import open_image, auto_map_channels

import Sequential_Fish.pipeline.tools._folder_integrity as prepro
import pandas as pd
import os
import re


#Reading input folder.
file_dict = prepro.assert_run_folder_integrity(
    run_path=RUN_PATH,
    fish_folder=FOLDER_KEYS.get('fish'),
    nucleus_folder=FOLDER_KEYS.get('nucleus')
    )
location_list = list(file_dict.keys())
location_list.sort()
print("{0} locations found.".format(len(location_list)))

#Init pandas DF
COLUMNS = [
    "acquisition_id",
    "location",
    "cycle",
    "full_path",
    "fish_shape",
    "fish_map",
    "dapi_full_path",
    "dapi_shape",
    "dapi_map"
    ]
Acquisition = pd.DataFrame(columns=COLUMNS)
cycle_map = pd.read_excel(RUN_PATH + '/' + MAP_FILENAME)
## TO REMOVE
cycle_map = cycle_map.drop(21)
print(cycle_map)
color_number = len(GENES_NAMES_KEY)
cycle_number = len(cycle_map)


file_index = 0
for location_index, location in enumerate(location_list) :
    
    #Get dapi_path
    dapi_full_path = RUN_PATH + "/{0}/{1}/".format(FOLDER_KEYS.get('nucleus'), location)
    assert len(os.listdir(dapi_full_path)) == 1
    dapi_full_path += os.listdir(dapi_full_path)[0]
    assert os.path.isfile(dapi_full_path)
    dapi_im = open_image(dapi_full_path)
    dapi_shape = dapi_im.shape
    dapi_map = auto_map_channels(dapi_im, color_number=color_number, cycle_number=cycle_number, bead_channel=HAS_BEAD_CHANNEL)
    print("dapi_map : ", dapi_map)
    
    #Get fish_path
    fish_path = RUN_PATH + "/{0}/{1}/".format(FOLDER_KEYS.get('fish'), location)
    fish_path_list = os.listdir(fish_path)
    fish_path_list.sort() # We sort so first file is main multi-tiff file.
    fish_im = open_image(fish_path + fish_path_list[0]) #Opening first tiff file will open all tiff files of this location (multitif_file) with correct reshaping. Ignoring first dim which will be the cycles gives us image dimension
    fish_shape = fish_im.shape[1:] #Opening first tiff file will open all tiff files of this location (multitif_file) with correct reshaping. Ignoring first dim which will be the cycles gives us image dimension
    fish_map = auto_map_channels(fish_im, color_number=color_number, cycle_number=cycle_number, bead_channel=HAS_BEAD_CHANNEL)
    print("fish_map : ", fish_map)
    for file in fish_path_list :
        cycle = int(re.findall(cycle_regex, file)[0])
        fish_full_path = fish_path + file
        assert os.path.isfile(dapi_full_path)

        Acquisition.loc[file_index] = [file_index, location, cycle, fish_full_path, fish_shape, fish_map, dapi_full_path, dapi_shape, dapi_map]
        file_index += 1

#Integrity checks
assert all(Acquisition['cycle'].isin(cycle_map[CYCLE_KEY])), "Some cycle are not found in map"
assert len(cycle_map[CYCLE_KEY] == len(Acquisition['cycle'])), "{0} column length doesn't match cycle number ({1})".format(CYCLE_KEY, len(Acquisition['cycle']))
for key in GENES_NAMES_KEY : 
    assert len(cycle_map[key] == len(Acquisition['cycle'])), "{0} column length doesn't match cycle number ({1})".format(key, len(Acquisition['cycle']))

Acquisition = pd.merge(
    left=Acquisition,
    right=cycle_map,
    left_on='cycle',
    right_on=CYCLE_KEY
).sort_values('acquisition_id').reset_index(drop=True)


map_dict ={"cycle" : list(cycle_map[CYCLE_KEY])}
map_dict.update({
    "{0}".format(gene_number) : list(cycle_map[gene_key]) for gene_number, gene_key in enumerate(GENES_NAMES_KEY)
})

color_columns = ["{0}".format(gene_number) for gene_number, gene_key in enumerate(GENES_NAMES_KEY)]
Gene_map = pd.DataFrame(map_dict)
Gene_map = Gene_map.melt(
    id_vars=['cycle'],
    value_vars=color_columns,
    value_name= "target",
    var_name="color_id"
)
Gene_map =Gene_map.reset_index(drop=False, names="map_id")
washout_index = Gene_map[Gene_map['target'] == WASHOUT_KEY_WORD].index
Gene_map.loc[washout_index, ['target']] = Gene_map.loc[washout_index]['target'] + '_' + Gene_map.loc[washout_index]['cycle'].astype(str) + '_' + Gene_map.loc[washout_index]['color_id'].astype(str)
assert len(Gene_map['target']) == len(Gene_map['target'].unique()), "{1} duplicates found in Gene map even after washout renaming... If several cycle targets same RNA please add suffix in Gene map to differenciate.\nFound genes : \n{0}".format(Gene_map['target'], len(Gene_map['target']) - len(Gene_map['target'].unique()))


#Output
save_path = RUN_PATH + '/result_tables/'
os.makedirs(save_path, exist_ok=True)
Acquisition.to_excel(save_path + '/Acquisition.xlsx')
Acquisition.to_feather(save_path + '/Acquisition.feather')
Gene_map.to_excel(save_path + 'Gene_map.xlsx')
Gene_map.to_feather(save_path + 'Gene_map.feather')
print("Done")