"""
This script aims at correcting drift computed in Drift.py for spots and clusters saving both corrected coordinates and detected coordinates.
"""
import pandas as pd
from Sequential_Fish.pipeline.tools import safe_merge_no_duplicates

from Sequential_Fish.pipeline_parameters import RUN_PATH 

Acquisition = pd.read_feather(RUN_PATH + '/result_tables/Acquisition.feather')
Drift = pd.read_feather(RUN_PATH + '/result_tables/Drift.feather')
Spots = pd.read_feather(RUN_PATH + '/result_tables/Spots.feather')
Clusters = pd.read_feather(RUN_PATH + '/result_tables/Clusters.feather')
Detection = pd.read_feather(RUN_PATH + '/result_tables/Detection.feather')

### Adding columns to Spots & Clusters
Spots = safe_merge_no_duplicates(Spots, Detection, 'acquisition_id', on = 'detection_id')
Spots = safe_merge_no_duplicates(Spots, Drift.loc[Drift['drift_type'] == 'fish'], 'acquisition_id', on = ['drift_z','drift_y','drift_x'])
Spots = safe_merge_no_duplicates(Spots, Acquisition, 'acquisition_id', on = 'fish_reodered_shape')

z_shape,y_shape,x_shape,_ = zip(*list(Spots['fish_reodered_shape']))
Spots['z_shape'] = z_shape
Spots['y_shape'] = y_shape
Spots['x_shape'] = x_shape
Spots = Spots.drop(columns='fish_reodered_shape')

Clusters = safe_merge_no_duplicates(Clusters, Detection, 'acquisition_id', on = 'detection_id')
Clusters = safe_merge_no_duplicates(Clusters, Drift.loc[Drift['drift_type'] == 'fish'], 'acquisition_id', on = ['drift_z','drift_y','drift_x'])
Clusters = safe_merge_no_duplicates(Clusters, Acquisition, 'acquisition_id', on = 'fish_reodered_shape')

if len(Clusters) > 0 :
    z_shape,y_shape,x_shape,_ = zip(*list(Clusters['fish_reodered_shape']))
    Clusters['z_shape'] = z_shape
    Clusters['y_shape'] = y_shape
    Clusters['x_shape'] = x_shape
if 'fish_reodered_shape' in Clusters.columns : Clusters = Clusters.drop(columns='fish_reodered_shape')

### Adding locations to Drift & Detection
Drift = safe_merge_no_duplicates(Drift, Acquisition, keys= 'location', on='acquisition_id')
Detection = safe_merge_no_duplicates(Detection, Acquisition, keys= 'location', on='acquisition_id')

#Applying drift correction
for key in ['drifted_z', 'drifted_y', 'drifted_x'] :
    if key in Spots.columns : Spots = Spots.drop(columns=key)
    if key in Clusters.columns : Clusters = Clusters.drop(columns=key)
Spots = Spots.rename(columns={'z' : 'drifted_z', 'y' : 'drifted_y', 'x' : 'drifted_x'}) #Keeping old values
for dim_index, i in enumerate(['z','y','x']) :
    Spots[i] = (Spots['drifted_{0}'.format(i)] + Spots['drift_{0}'.format(i)]).astype(int)
    drop_index = Spots[Spots[i] >= Spots['{0}_shape'.format(i)]].index
    Spots = Spots.drop(drop_index)
    print("drift pushed {0} spots out of range".format(len(drop_index)))

Clusters = Clusters.rename(columns={'z' : 'drifted_z', 'y' : 'drifted_y', 'x' : 'drifted_x'}) #Keeping old values
for dim_index, i in enumerate(['z','y','x']) : 
    Clusters[i] = (Clusters['drifted_{0}'.format(i)] + Clusters['drift_{0}'.format(i)]).astype(int)
    drop_index = Clusters[Clusters[i] >= Clusters['{0}_shape'.format(i)]].index
    print("drift pushed {0} clusters out of range".format(len(drop_index)))
    Clusters = Clusters.drop(drop_index)

Spots.reset_index(drop=True).to_feather(RUN_PATH + "/result_tables/Spots.feather")
Clusters.reset_index(drop=True).to_feather(RUN_PATH + "/result_tables/Clusters.feather")