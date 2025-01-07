"""
Aims at finding drift value for each field of view and store it into a dataframe : Drift
"""

import os
import pandas as pd
import numpy as np
import pbwrap.preprocessing.alignement as prepro
import pbwrap.detection as detection
from pbwrap.utils import open_image
from tqdm import tqdm

from Sequential_Fish.pipeline_parameters import RUN_PATH, FISH_PENALTY, DAPI_PENALTY, SLICE_TO_REMOVE, VOXEL_SIZE, BEAD_SIZE

SAVE_PATH = RUN_PATH + '/visuals/'
Drift_columns = [
    'acquisition_id',
    'drift_type',
    'drift_z',
    'drift_y',
    'drift_x',
    'ref_bead_threshold',
    'drift_bead_threshold',
    'ref_bead_number',
    'drift_bead_number',
    'found_symetric_drift',
    'voxel_size',
    'bead_size',
]
Drift_save = pd.DataFrame(columns=Drift_columns)
Drift_save['found_symetric_drift'] = Drift_save['found_symetric_drift'].astype(bool)

### MAIN ###

Acquisition = pd.read_feather(RUN_PATH + "/result_tables/Acquisition.feather")



reference_threshold_penalty = 1
drift_treshold_penalty = 1

bug = 0

for location in Acquisition['location'].unique() : 
    
    print('Starting ',location)
    plot_path = SAVE_PATH + '/drift/{0}/'.format(location)
    os.makedirs(plot_path,exist_ok=True)
    #
    print("opening images...")
    sub_acq = Acquisition.loc[Acquisition["location"] == location].sort_values('cycle')
    dapi_path = sub_acq['dapi_full_path'].unique()
    assert len(dapi_path) == 1, 'multiple files for dapi found : {0}'.format(len(dapi_path))
    dapi_path = dapi_path[0]

    fish_path = sub_acq.loc[sub_acq['cycle']==0]['full_path']
    assert len(fish_path) == 1, 'multiple files for fish found : {0}'.format(len(fish_path))
    fish_path = fish_path.iat[0]

    dapi_image = open_image(dapi_path)
    fish_image_stack = open_image(fish_path)

    max_z = max(
        dapi_image.shape[0],
        fish_image_stack.shape[1]
    )
    
    #Selecting images
    fish_image_stack = fish_image_stack[...,-1] # Selecting beads channel
    assert len(sub_acq) == len(fish_image_stack) == len(sub_acq['acquisition_id'])
    fish_image_stack = fish_image_stack[:,SLICE_TO_REMOVE:max_z,...] # removing first slice that is noisy
    fish_reference_image = fish_image_stack[0]
    fish_other_images = fish_image_stack[1:]
    dapi_image = dapi_image[SLICE_TO_REMOVE:max_z,-1] #To check
    assert dapi_image.shape[-1] > 100, "beads selected in wrong channel, remaining dapi shape : {0}".format(dapi_image.shape)

    #Reference fov has no drift
    ref_acquisition_id = sub_acq[sub_acq['cycle'] == 0]['acquisition_id'].iat[0]
    Drift = pd.DataFrame({
        'acquisition_id' : [ref_acquisition_id],
        'drift_type' : ['fish'],
        'drift_z' : [0],
        'drift_y' : [0],
        'drift_x' : [0],
        'ref_bead_threshold' : [np.NaN],
        'drift_bead_threshold' : [np.NaN],
        'ref_bead_number' : [np.NaN],
        'drift_bead_number' : [np.NaN],
        'found_symetric_drift' : [True],
    })

    #preparing fish loop
    stack_index = 0
    ref_threshold = None
    drift_threshold = None


    print("Detecting beads for reference stack...")
    reference_beads, ref_threshold = detection.detect_spots(
        images= fish_reference_image,
        threshold= ref_threshold,
        threshold_penalty= FISH_PENALTY,
        voxel_size= VOXEL_SIZE,
        spot_radius= BEAD_SIZE,
        return_threshold=True,
        ndim=3,
    )

    

    #dapi drift
    dapi_results = prepro.find_drift(
        reference_bead_signal= fish_reference_image,
        drifted_bead_signal= dapi_image,
        voxel_size= VOXEL_SIZE,
        bead_size= BEAD_SIZE,
        reference_threshold= ref_threshold,
        drift_threshold_penalty= DAPI_PENALTY,
        plot_path= plot_path + 'dapi_',
        extended_result = True
    )
    dapi_results = pd.DataFrame(dapi_results)
    dapi_results['acquisition_id'] = ref_acquisition_id
    dapi_results['drift_type'] = 'dapi'

    Drift = pd.concat([
        Drift,
        dapi_results
    ], axis=0)

    print("Detecting beads & computing drift values for drifted fish stack...")
    for acquisition_id in tqdm(sub_acq[sub_acq['cycle'] != 0]['acquisition_id']) : #is ordered by cycle which is image stack ordered.

        drifted_beads, drift_threshold = detection.detect_spots(
            images= fish_other_images[stack_index],
            threshold= drift_threshold,
            threshold_penalty= FISH_PENALTY,
            voxel_size= VOXEL_SIZE,
            spot_radius= BEAD_SIZE,
            return_threshold=True,
            ndim=3
        )

        ref_bead_number = len(reference_beads)
        drift_bead_number = len(drifted_beads)

        #Finding threshold to consider beads are matching
        shape = np.max([fish_reference_image.shape, fish_other_images[stack_index].shape], axis=0)
        reference_distance_map, reference_indices = prepro._build_maps(spots= reference_beads, voxel_size=VOXEL_SIZE, shape=shape)
        drifted_distance_map, drifted_indices = prepro._build_maps(spots=drifted_beads, voxel_size=VOXEL_SIZE, shape=shape)
        distance_reference_from_drift = prepro._get_distance_from_map(spots= reference_beads, distance_map= drifted_distance_map)
        distance_drift_from_reference = prepro._get_distance_from_map(spots= drifted_beads, distance_map= reference_distance_map)
        distance_threshold = prepro._find_distance_threshold(
            distance_reference_from_drift, 
            distance_drift_from_reference,
            output_path= plot_path + 'fish_{0}_'.format(acquisition_id),
            )

        coordinates_df = prepro._build_coordinates_df(
             reference_beads,
             drifted_beads,
             reference_indices,
             drifted_indices,
             distance_reference_from_drift,
             distance_drift_from_reference,
             dim=3
            )

        coordinates_df = prepro._apply_distance_threshold(
            coordinates_df,
            distance_threshold
            )
        
        try :
            reference_drift, drift = prepro._find_drift_value(
                matching_coordinates_df=coordinates_df.dropna(axis=0),
                path_output= plot_path + 'fish_{0}_'.format(acquisition_id),
                )
        
        except ValueError as e :
            print("{}".format(e))
            print("Adding NaN values for this acquisition")
            bug+=1
            os.makedirs(RUN_PATH + '/bugs/',exist_ok=True)
            coordinates_df.reset_index(drop=False).to_feather(RUN_PATH + '/bugs/coordinates_df_{0}'.format(bug))

            drift = [np.NaN, np.NaN, np.NaN]
            found_sym_drift = np.NaN
        
        else :
            found_sym_drift = all(np.array(reference_drift) == -np.array(drift))

        Drift = pd.concat([
            Drift,
            pd.DataFrame({
                'acquisition_id' : [acquisition_id],
                'drift_type' : ['fish'],
                'drift_z' : [drift[0]],
                'drift_y' : [drift[1]],
                'drift_x' : [drift[2]],
                'ref_bead_threshold' : [ref_threshold],
                'drift_bead_threshold' : [drift_threshold],
                'ref_bead_number' : [ref_bead_number],
                'drift_bead_number' : [drift_bead_number],
                'found_symetric_drift' : [found_sym_drift]
            })
        ], axis=0)

    Drift_save = pd.concat([
        Drift_save,
        Drift
    ], axis=0)
    stack_index +=1

print("All locations computed. Saving results...")

Drift_save['voxel_size'] = [VOXEL_SIZE] * len(Drift_save)
Drift_save['bead_size'] = [BEAD_SIZE] * len(Drift_save)
Drift_save = Drift_save.reset_index(drop=True).reset_index(drop=False, names= 'drift_id')
Drift_save.to_feather(RUN_PATH + '/result_tables/Drift.feather')
Drift_save.to_excel(RUN_PATH + '/result_tables/Drift.xlsx')

print("Done")