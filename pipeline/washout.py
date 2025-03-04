"""
This script aims at removing spots found in washout. 
If a spot is detected during a washout cycle, all spots detected in succeeding cycles at same location are deleted.
"""

import pandas as pd
import numpy as np

from Sequential_Fish.pipeline_parameters import RUN_PATH, WASHOUT_KEY_WORD
from Sequential_Fish.pipeline.tools import safe_merge_no_duplicates


Acquisition = pd.read_feather(RUN_PATH + '/result_tables/Acquisition.feather')
Detection = pd.read_feather(RUN_PATH + '/result_tables/Detection.feather')
Clusters = pd.read_feather(RUN_PATH + '/result_tables/Clusters.feather')
Spots = pd.read_feather(RUN_PATH + '/result_tables/Spots.feather')
Gene_map = pd.read_feather(RUN_PATH + '/result_tables/Gene_map.feather')

Gene_map['is_washout'] = Gene_map['target'].str.contains(WASHOUT_KEY_WORD)


# Joins
Detection = safe_merge_no_duplicates(
    Detection,
    Acquisition,
    keys= ['cycle'],
    on= 'acquisition_id'
)
Spots = safe_merge_no_duplicates(
    Spots,
    Detection,
    keys=['cycle','color_id'],
    on= 'detection_id'
)
Spots = safe_merge_no_duplicates(
    Spots,
    Gene_map,
    keys='is_washout',
    on= ['cycle','color_id']
)

# putting coordinates in tuples
Spots['coordinates'] = list(zip(Spots['z'], Spots['y'], Spots['x']))

#Filtering
cycle_list = list(Spots['cycle'].unique())
cycle_list.sort()

banned_coordinates = []
for cycle in cycle_list :

    new_washout_idx = Spots.loc[
        (Spots['coordinates'].isin(banned_coordinates)) & (Spots['cycle'] == cycle)
        ].index

    Spots.loc[new_washout_idx, ['is_washout']] = True

    new_banned_coordinates = list(
        Spots[
            (Spots['is_washout']) & (Spots['cycle'] == cycle)
            ]['coordinates'].unique()
    )

    #Updating banned coordinates
    banned_coordinates += new_banned_coordinates
    banned_coordinates = list(pd.unique(banned_coordinates))