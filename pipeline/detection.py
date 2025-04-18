
"""
Spot quantification for Sequential Fish data

This script use results from FishSeq_pipeline_segmentation.py that must be run before
"""

import os, sys
import warnings
import numpy as np
import pandas as pd
from Sequential_Fish.tools import open_image, reorder_image_stack
from concurrent.futures import ThreadPoolExecutor
from pbwrap.detection.multithread import multi_thread_full_detection, build_Spots_and_Cluster_df
from tqdm import tqdm

#########
## USER PARAMETERS
#########

def main(run_path) :

    print(f"detection runing for {run_path}")
    
    if len(sys.argv) == 1:
        from Sequential_Fish.pipeline_parameters import detection_MAX_WORKERS as MAX_WORKERS
        from Sequential_Fish.pipeline_parameters import VOXEL_SIZE, SPOT_SIZE, ALPHA, BETA, GAMMA, CLUSTER_SIZE, MIN_SPOT_PER_CLUSTER, ARTIFACT_RADIUS, DETECTION_SLICE_TO_REMOVE
    
    else :
        from Sequential_Fish.run_saves import get_parameter_dict
        PARAMETERS = ['VOXEL_SIZE', 'SPOT_SIZE', 'ALPHA', 'BETA', 'GAMMA', 'CLUSTER_SIZE', 'MIN_SPOT_PER_CLUSTER', 'ARTIFACT_RADIUS', 'DETECTION_SLICE_TO_REMOVE', 'detection_MAX_WORKERS']
        
        parameters_dict = get_parameter_dict(run_path, PARAMETERS)
        
        VOXEL_SIZE = parameters_dict['VOXEL_SIZE']
        SPOT_SIZE = parameters_dict['SPOT_SIZE']
        ALPHA = parameters_dict['ALPHA']
        BETA = parameters_dict['BETA']
        GAMMA = parameters_dict['GAMMA']
        CLUSTER_SIZE = parameters_dict['CLUSTER_SIZE']
        MIN_SPOT_PER_CLUSTER = parameters_dict['MIN_SPOT_PER_CLUSTER']
        ARTIFACT_RADIUS = parameters_dict['ARTIFACT_RADIUS']
        DETECTION_SLICE_TO_REMOVE = parameters_dict['DETECTION_SLICE_TO_REMOVE']
        MAX_WORKERS = parameters_dict['detection_MAX_WORKERS']
    
    #Loading data
    Acquisition = pd.read_feather(run_path + "/result_tables/Acquisition.feather")

    #preparing folders
    save_path = run_path + "/result_tables/"

    #Main loop
    Detection_save = pd.DataFrame()
    Spots_save = pd.DataFrame()
    Clusters_save = pd.DataFrame()

    os.makedirs(run_path + '/detection_fov/',exist_ok=True)

    max_id = 0
    for location_id, location in enumerate(Acquisition['location'].unique()) :
        print("Starting location {0}...".format(location_id))
        sub_data = Acquisition.loc[Acquisition["location"] == location]

        visual_path = run_path + "/visuals/{0}/".format(location)
        os.makedirs(visual_path, exist_ok=True)

        #Opening images
        print("loading images...")
        image_path = sub_data[sub_data['cycle'] == 0]['full_path']
        image_map = sub_data[sub_data['cycle'] == 0]['fish_map']
        assert len(image_path) == 1, image_path
        image_path = image_path.iat[0]    
        image_map = image_map.iat[0]    
        multichannel_stack = open_image(image_path)# This open 4D multichannel image (all the images are loaded in one call)
        multichannel_stack = reorder_image_stack(multichannel_stack, image_map)

        #Converting na back to None
        bottom_index, top_index = DETECTION_SLICE_TO_REMOVE
        if bottom_index is None :
            pass
        elif np.isnan(bottom_index) : 
            bottom_index = None
        elif type(bottom_index) != int : bottom_index= int(bottom_index)
        
        if top_index is None :
            pass
        elif np.isnan(top_index) : 
            top_index = None
        elif type(top_index) != int : 
            top_index= int(bottom_index)
        
        
        #Removing Z slices (USER SETTING)
        if type(top_index) != type(None) : top_index = -top_index
        
        multichannel_stack = multichannel_stack[:,bottom_index:top_index]

        multichannel_stack = multichannel_stack[...,:-1]
        images_list = [np.moveaxis(channel,[3,0,1,2],[0,1,2,3]) for channel in multichannel_stack]
        images_list = [
            [colors for colors in channel]
             for channel in images_list]
        image_number = len(multichannel_stack)
        colors = list(zip(*images_list))
        colors_number = len(colors)

        #Preparing threads arguments
        Detection = pd.DataFrame({
            'acquisition_id' : list(sub_data['acquisition_id'])
            ,'visual_name' : [visual_path] * image_number
            ,'filename' : list(sub_data['full_path'])
            ,'voxel_size' : [tuple(VOXEL_SIZE)] * image_number
            ,'spot_size' : [tuple(SPOT_SIZE)] * image_number
            ,'alpha' : [ALPHA] * image_number
            ,'beta' : [BETA] * image_number
            ,'gamma' : [GAMMA] * image_number
            ,'artifact_radius' : [ARTIFACT_RADIUS] * image_number
            ,'cluster_size' : [CLUSTER_SIZE] * image_number
            ,'min_spot_per_cluster' : [MIN_SPOT_PER_CLUSTER] * image_number
        })

        if 'threshold' in Acquisition.columns : Acquisition = Acquisition.drop(columns='threshold')
        threshold_col_mask = Acquisition.columns.str.contains('Threshold') #looking for Threshold_0, Threshold_1 col
        if threshold_col_mask.any() :
            threshold_col = Acquisition.columns[threshold_col_mask]
            Detection_line_number = len(Detection)
            Acquisition.loc[:,threshold_col] = Acquisition.loc[:,threshold_col].fillna('').replace('',None)
            Detection = pd.merge(
                left= Detection,
                right= Acquisition.loc[:,['acquisition_id'] + list(Acquisition.columns[threshold_col_mask])],
                how='inner'
            )
            assert len(Detection) == Detection_line_number, "Duplicates or missing acquisition id in Acquisition df."

            Detection['threshold'] = None # above added threshold are like Threshold_i with i the color_number (0,1...)

        else :
            print("threshold column not found in Acquisition, automatic threshold will be used.")
            Detection['threshold'] = None

        id_columns = Detection.columns

        colors_columns = []
        for color_num, color in enumerate(colors) :
            colors_columns.append('{0}'.format(color_num))
            Detection['{0}'.format(color_num)] = color
        id_columns = list(id_columns)
        Detection = Detection.melt(
            id_vars= id_columns,
            value_vars= colors_columns,
            var_name= "color_id",
            value_name= "image",
        )
        Detection['visual_name'] = Detection['visual_name'] + Detection['acquisition_id'].astype(str) + Detection['color_id'].astype(str)
        Detection = Detection.reset_index(drop=False, names='detection_id')
        Detection['detection_id'] += max_id +1
        max_id = Detection['detection_id'].max()

        for color_id in Detection['color_id'].unique() :
            target = 'Threshold_{0}'.format(color_id)
            if target in Detection.columns : 
                loc_index = Detection.loc[Detection['color_id'] == color_id].index
                Detection.loc[loc_index,['threshold']] = Detection[target]

        #Launching threads
        with ThreadPoolExecutor(max_workers= MAX_WORKERS) as executor :
            detection_result = tqdm(executor.map(
                multi_thread_full_detection,
                Detection['image'],
                Detection['voxel_size'],
                Detection['threshold'],
                Detection['spot_size'],
                Detection['alpha'],
                Detection['beta'],
                Detection['gamma'],
                Detection['artifact_radius'],
                Detection['cluster_size'],
                Detection['min_spot_per_cluster'],
                Detection['visual_name'],
                Detection['detection_id'],
            ))
        Spots, Clusters = build_Spots_and_Cluster_df(detection_result)

        #Correct coordinates for removed slices
        if type(DETECTION_SLICE_TO_REMOVE[0]) != type(None) :
            Spots['z'] = Spots['z'] + DETECTION_SLICE_TO_REMOVE[0]
            Clusters['z'] = Clusters['z'] + DETECTION_SLICE_TO_REMOVE[0]

        #Saving detection field view
        print("Saving field of views as compressed arrays.")
        save_dict = {
            str(detection_id) : np.max(image,axis=0) for detection_id,image in zip(Detection['detection_id'],Detection['image'])
        }

        np.savez(
            run_path + '/detection_fov/{0}'.format(location), #PATH,
            **save_dict
        )

        Detection['image_path'] = run_path + '/detection_fov/{0}.npz'.format(location)
        Detection['image_key'] = Detection['detection_id'].astype(str)
        Detection = Detection.drop(columns='image')

        #Appending dataframes
        Detection_save = pd.concat([
            Detection_save,
            Detection
            ], axis=0).reset_index(drop=True)

        Spots_save = pd.concat([
            Spots_save,
            Spots
            ], axis=0).reset_index(drop=True)

        Clusters_save = pd.concat([ 
            Clusters_save,  
            Clusters    
            ], axis=0).reset_index(drop=True)   
        ###### End For loop #####   

    #Unique Spots_identifier    
    Spots_save = Spots_save.drop(columns='spot_id').reset_index(drop=False, names="spot_id")

    #Saving results 
    Detection_save.to_feather(save_path + '/Detection.feather')
    Spots_save.to_feather(save_path + '/Spots.feather') 
    Clusters_save.to_feather(save_path + '/Clusters.feather')
    
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        warnings.warn("Prefer launching this script with command : 'python -m Sequential_Fish pipeline detection' or make sure there is no conflict for parameters loading in pipeline_parameters.py")
        from Sequential_Fish.pipeline_parameters import RUN_PATH as run_path
    else :
        run_path = sys.argv[1]
    main(run_path)     