"""
This script aims at reading the input folder and preparing data folders and locations for next scripts.
"""
import sys
import Sequential_Fish.tools._folder_integrity as prepro
import pandas as pd
import os
import warnings
import numpy as np
from Sequential_Fish.tools.utils import open_image, auto_map_channels, _find_one_or_NaN, reorder_image_stack

def main(run_path) :

    print(f"input runing for {run_path}")

    if len(sys.argv) == 1:
        from Sequential_Fish.pipeline_parameters import FOLDER_KEYS, MAP_FILENAME, cycle_regex, CYCLE_KEY, GENES_NAMES_KEY, WASHOUT_KEY_WORD, HAS_BEAD_CHANNEL
    else :
        from Sequential_Fish.run_saves import get_parameter_dict
        PARAMETERS = ['nucleus_folder','fish_folder', 'MAP_FILENAME', 'cycle_regex', 'CYCLE_KEY', 'GENES_NAMES_KEY', 'WASHOUT_KEY_WORD', 'HAS_BEAD_CHANNEL']
        parameters_dict = get_parameter_dict(run_path, parameters=PARAMETERS)
        nucleus_folder = parameters_dict['nucleus_folder']
        fish_folder = parameters_dict['fish_folder']
        MAP_FILENAME = parameters_dict['MAP_FILENAME']
        cycle_regex = parameters_dict['cycle_regex']
        CYCLE_KEY = parameters_dict['CYCLE_KEY']
        GENES_NAMES_KEY = parameters_dict['GENES_NAMES_KEY']
        WASHOUT_KEY_WORD = parameters_dict['WASHOUT_KEY_WORD']
        HAS_BEAD_CHANNEL = parameters_dict['HAS_BEAD_CHANNEL']
        
        FOLDER_KEYS = {
            'nucleus_folder' : nucleus_folder,
            'fish_folder' : fish_folder,
        }
    
    #Reading input folder.
    file_dict = prepro.assert_run_folder_integrity(
        run_path=run_path,
        fish_folder=FOLDER_KEYS.get('fish_folder'),
        nucleus_folder=FOLDER_KEYS.get('nucleus_folder')
        )
    location_list = list(file_dict.keys())
    location_list.sort()
    location_number = len(location_list)
    print("{0} locations found.".format(location_number))

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
    cycle_map = pd.read_excel(run_path + '/' + MAP_FILENAME)
    color_number = len(GENES_NAMES_KEY)
    cycle_number = len(cycle_map)
    print("Expected {0} colors.".format(color_number))
    print("Expected {0} cycles.".format(cycle_number))

    file_index = 0
    Acquisition['acquisition_id'] = np.arange(len(location_list)*cycle_number)
    Acquisition['location'] = location_list * cycle_number
    cycles_list = list(cycle_map[CYCLE_KEY])*location_number
    cycles_list.sort()
    Acquisition['cycle'] = cycles_list
    for location_index, location in enumerate(location_list) :

        #Get dapi_path
        dapi_full_path = run_path + "/{0}/{1}/".format(FOLDER_KEYS.get('nucleus_folder'), location)
        assert len(os.listdir(dapi_full_path)) == 1
        dapi_full_path += os.listdir(dapi_full_path)[0]
        assert os.path.isfile(dapi_full_path)
        dapi_im = open_image(dapi_full_path)
        dapi_shape = dapi_im.shape
        dapi_map = auto_map_channels(dapi_im, color_number=1, cycle_number=cycle_number, bead_channel=HAS_BEAD_CHANNEL)
        dapi_reodered_shape = reorder_image_stack(dapi_im, dapi_map).shape

        #Setting dapi informations
        index = Acquisition[Acquisition['location'] == location].index
        Acquisition.loc[index, ['dapi_full_path']] = dapi_full_path
        Acquisition.loc[index, ['dapi_shape']] = pd.Series((tuple(dapi_shape),)*cycle_number, index=index)
        Acquisition.loc[index, ['dapi_map']] = pd.Series((dapi_map,)* cycle_number, index=index)

        #Setting general fish informations
        fish_path = run_path + "/{0}/{1}/".format(FOLDER_KEYS.get('fish_folder'), location)
        fish_path_list = os.listdir(fish_path)
        fish_path_list.sort() # THIS MUST GIVE CYCLE ORDERED LIST ie : filename cycle matches map cycles and rest of filename doesn't change list order.
        fish_im = open_image(fish_path + fish_path_list[0]) #Opening first tiff file will open all tiff files of this location (multitif_file) with correct reshaping. Ignoring first dim which will be the cycles gives us image dimension
        fish_map = auto_map_channels(fish_im, color_number=color_number, cycle_number=cycle_number, bead_channel=HAS_BEAD_CHANNEL)
        fish_shape = fish_im.shape[:fish_map['cycles']] + fish_im.shape[(fish_map['cycles'] + 1):] #1cycle per acquisition
        reoderdered_shape = reorder_image_stack(fish_im, fish_map).shape
        fish_reodered_shape = reoderdered_shape[1:]

        full_path_list = [fish_path + file for file in fish_path_list]
        while len(full_path_list) < len(index) :
            full_path_list.append(np.NaN)

        Acquisition.loc[index, "fish_shape"] = pd.Series((fish_shape,)*cycle_number, index=index)
        Acquisition.loc[index, "fish_map"] = pd.Series((fish_map,)*cycle_number, index=index)
        Acquisition.loc[index, "full_path"] = full_path_list
        Acquisition.loc[index, "fish_reodered_shape"] = pd.Series((fish_reodered_shape,)*cycle_number, index=index)

        cycle_regex_result = Acquisition.loc[:, 'full_path'].apply(_find_one_or_NaN, regex=cycle_regex)

    #Integrity checks
    assert all(Acquisition['cycle'].isin(cycle_map[CYCLE_KEY])), "Some cycle are not found in map"
    assert len(cycle_map[CYCLE_KEY] == len(Acquisition['cycle'])), "{0} column length doesn't match cycle number ({1})".format(CYCLE_KEY, len(Acquisition['cycle']))
    for key in GENES_NAMES_KEY : 
        assert len(cycle_map[key] == len(Acquisition['cycle'])), "{0} column length doesn't match cycle number ({1})".format(key, len(Acquisition['cycle']))

    cycle_regex_result = Acquisition.loc[:, 'full_path'].apply(_find_one_or_NaN, regex=cycle_regex)
    cycles_match = all(Acquisition.loc[~Acquisition['full_path'].isna(),"cycle"] == cycle_regex_result[~cycle_regex_result.isna()])
    if not cycles_match : raise ValueError("Missmatch between cycles assigned and cycles found in filenames. Maybe filenames could not be used to sort on cycles.")
    if any(Acquisition['full_path'].isna()) : warnings.warn("Warning : Some images registered in metadata were not found in folder. Ignore this message if some files were deleted after acquisition, in such a case pipeline should return as well 'OME series failed to read [...]. Missing data are zeroed' warning. ")

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
    save_path = run_path + '/result_tables/'
    os.makedirs(save_path, exist_ok=True)
    Acquisition.to_excel(save_path + '/Acquisition.xlsx')
    Acquisition.to_feather(save_path + '/Acquisition.feather')
    Gene_map.to_excel(save_path + 'Gene_map.xlsx')
    Gene_map.to_feather(save_path + 'Gene_map.feather')
    print("Done")
    
    
if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 1:
        warnings.warn("Prefer launching this script with command : 'python -m Sequential_Fish pipeline input' or make sure there is no conflict for parameters loading in pipeline_parameters.py")
        from Sequential_Fish.pipeline_parameters import RUN_PATH as run_path
    else :
        run_path = sys.argv[1]
    main(run_path)    
