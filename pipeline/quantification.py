import numpy as np
import pandas as pd
import CustomPandasFramework.Fish_seq as FishSeq
from bigfish.multistack import match_nuc_cell
from pbwrap.preprocessing import shift_array
from concurrent.futures import ThreadPoolExecutor
from pbwrap.detection.multithread import cell_quantification
from tqdm import tqdm

from Sequential_Fish.pipeline_parameters import quantif_MAX_WORKERS as MAX_WORKERS
from Sequential_Fish.pipeline_parameters import RUN_PATH, COLOC_POPULATION, VOXEL_SIZE, COLOC_DISTANCE

Acquisition = pd.read_feather(RUN_PATH + '/result_tables/Acquisition.feather')
Drift = pd.read_feather(RUN_PATH + '/result_tables/Drift.feather')
# Abberation = pd.read_feather(RUN_PATH + '/result_tables/Aberration.feather') #TODO
Spots = pd.read_feather(RUN_PATH + '/result_tables/Spots.feather')
Clusters = pd.read_feather(RUN_PATH + '/result_tables/Clusters.feather')
Detection = pd.read_feather(RUN_PATH + '/result_tables/Detection.feather')

# Matching location with Drift
Drift_len = len(Drift)
if not 'location' in Drift.columns :
    Drift = pd.merge(
        Drift,
        Acquisition.loc[:,['acquisition_id', 'location']],
        on= 'acquisition_id',
        how= 'inner'
    )
assert len(Drift) == Drift_len, "Duplicate or deletion during Drift/Acquisition merge."

#Matching location with Detection
Detection_len = len(Detection)
if 'location' not in Detection.columns :
    Detection = pd.merge(
        Detection,
        Acquisition.loc[:,['acquisition_id', 'location']],
        on= 'acquisition_id',
        how= 'inner'
    )
    assert len(Detection) == Detection_len, "Duplicate or deletion during Detection/Acquisition merge."


Cell_save = pd.DataFrame()
Colocalisation_save =pd.DataFrame()
for location in Acquisition['location'].unique() :
    print("Starting {0}".format(location))
    segmentation_results = np.load(RUN_PATH + '/segmentation/{0}_segmentation.npz'.format(location))
    cytoplasm_label = segmentation_results['cytoplasm']
    nucleus_label = segmentation_results['nucleus']
    dapi_signal = segmentation_results['dapi_signal']

    #Correct drift for nucleus
    drift = Drift.loc[(Drift['drift_type'] == 'dapi') & (Drift['location'] == location), ['drift_y', 'drift_x']].to_numpy(dtype=int).squeeze()
    nucleus_label = shift_array(nucleus_label, *drift)

    #Correct abberation for nucleus
    #TODO

    nucleus_label, cytoplasm_label = match_nuc_cell(nucleus_label, cytoplasm_label, single_nuc=True, cell_alone=False)
    #Getting Detection ids for this fov
    sub_Detection = Detection.loc[Detection['location'] == location]
    selected_detection_id = sub_Detection['detection_id']
    
    #Select all spots belonging to this fov; one list element per (cycle,color)
    all_fov_spots_lists = [
        np.array(list(
            zip(
                Spots[Spots['detection_id'] == detection_id]['z'],
                Spots[Spots['detection_id'] == detection_id]['y'],
                Spots[Spots['detection_id'] == detection_id]['x'],
            )
        ), dtype=int)
    for detection_id in selected_detection_id]
    #Select all clusters belonging to this fov; one list element per (cycle,color)
    all_fov_clusters_lists = [
        np.array(list(
            zip(
                Clusters[Clusters['detection_id'] == detection_id]['z'],
                Clusters[Clusters['detection_id'] == detection_id]['y'],
                Clusters[Clusters['detection_id'] == detection_id]['x'],
                Clusters[Clusters['detection_id'] == detection_id]['spot_number'],
                Clusters[Clusters['detection_id'] == detection_id]['cluster_id'].fillna(-1), #For bigfish compatibility
            )
        ))
    for detection_id in selected_detection_id]
    detection_number = len(selected_detection_id)

    #Launching threads on cell features
    detection_fov = np.load(RUN_PATH + '/detection_fov/{0}.npz'.format(location))
    fov_list = [detection_fov[fov_idx] for fov_idx in sub_Detection['image_key']] #TODO remove max projection once arrays will be saved directly in 2D


    print("Starting individual cell metrics for {0} detections".format(len(sub_Detection)))
    with ThreadPoolExecutor(max_workers= MAX_WORKERS) as executor :
        cell_quantification_result = list(tqdm(executor.map(
            cell_quantification,
            sub_Detection['acquisition_id'],
            selected_detection_id,
            all_fov_spots_lists,
            all_fov_clusters_lists,
            sub_Detection['voxel_size'],
            [cytoplasm_label] * detection_number,
            [nucleus_label] * detection_number,
            fov_list,
            [dapi_signal]*detection_number
        ),total= len(sub_Detection), desc="individual cell metrics"))
    Cell = pd.concat(cell_quantification_result, axis=0) 
    Cell['location'] = location


    #Compute colocalisation
    Colocalisation = FishSeq.compute_colocalisation(
        Spots=Spots,
        detection_id_list=selected_detection_id,
        population=COLOC_POPULATION,
        voxel_size=VOXEL_SIZE,
        coloc_distance=COLOC_DISTANCE,
        cell_label=cytoplasm_label,
        location=location,
        z_shape=Spots['z'].max(),
        max_workers=MAX_WORKERS,
    )


    ## End of loop
    Cell_save = pd.concat([
        Cell_save,
        Cell
    ],axis=0)

    Colocalisation_save = pd.concat([
        Colocalisation_save,
        Colocalisation
    ],axis=0)
    ####

#Building cell_id (not unique identifier bc Cell actually has one line per detection)
cell_IDs = Cell_save.groupby(['location','label'])['detection_id'].first().reset_index(drop=False).reset_index(drop=False, names='cell_id')
cell_len = len(Cell_save)
Cell_merged = pd.merge(
    left= Cell_save,
    right= cell_IDs.loc[:,['location','label','cell_id']],
    on= ('location','label'),
)
if len(Cell_merged) != cell_len : 
    print("\033[33mWARNING : Cell line conservation failed during merge. Saving a copy of the table before merge.\033[00m")
    Cell_save.reset_index(drop=True).to_feather(RUN_PATH = "/result_tables/Cell_before_merge.feather")
else :
    del Cell_save


#Building reference between Colocalisation and Cell tables and transmiting total rna number to all coloc lines
Cell['detection_id'] = Cell['detection_id'].astype(int)
Colocalisation_save['detection_id1'] = Colocalisation_save['detection_id1'].astype(int)
Colocalisation_save['detection_id2'] = Colocalisation_save['detection_id2'].astype(int)
coloc_len = len(Colocalisation_save)
Colocalisation_merged = Colocalisation_save.rename(columns={"object1" : "detection_id1", "object2" : "detection_id2"})

Colocalisation_merged:pd.DataFrame = pd.merge(
    Colocalisation_merged,
    Cell_merged.loc[:,['label','detection_id','rna_number']],
    right_on= ['label','detection_id'],
    left_on= ['label','detection_id1'],
).rename(columns={'rna_number' : 'spot1_total_number'})

Colocalisation_merged:pd.DataFrame = pd.merge(
    Colocalisation_merged,
    Cell_merged.loc[:,['label','detection_id','rna_number']],
    right_on= ['label','detection_id'],
    left_on= ['label','detection_id2'],
).rename(columns={'rna_number' : 'spot2_total_number'})

if len(Colocalisation_merged) != coloc_len : 
    print("\033[33mWARNING : Colocalisation line conservation failed during merge. Saving a copy of the table before merge.\nbefore : {0} ; after : {1}\033[00m".format(coloc_len, len(Colocalisation_merged)))
    Colocalisation_save.reset_index(drop=True).to_feather(RUN_PATH + "/result_tables/Coloc_before_merge.feather")
else :
    del Colocalisation_save

#Compute colocalisation fraction
Colocalisation_merged['fraction'] = Colocalisation_merged['count'] / Colocalisation_merged['spot1_total_number']
Colocalisation_merged['sub_fraction'] = Colocalisation_merged['count'] / Colocalisation_merged['spot1_number'] #sub fraction = fraction when population is all or when all spots are belongs to one population

#Save tables
Colocalisation_merged = Colocalisation_merged.reset_index(drop=True).reset_index(drop=False, names='colocalisation_id')
Cell_merged = Cell_merged.reset_index(drop=True).reset_index(drop=False, names='quantification_id')
Colocalisation_merged.to_feather(RUN_PATH + "/result_tables/Colocalisation.feather")
Cell_merged.to_feather(RUN_PATH + "/result_tables/Cell.feather")
Drift.reset_index(drop=True).to_feather(RUN_PATH + "/result_tables/Drift.feather")
Detection.reset_index(drop=True).to_feather(RUN_PATH + "/result_tables/Detection.feather")