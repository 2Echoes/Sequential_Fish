"""
Main script to call for analysis pipeline.
"""

import os
import pandas as pd

from .post_processing import Spots_filtering

from .density import density_analysis
from .distributions import distributions_analysis
from ..pipeline_parameters import RUN_PATH
from ..tools import safe_merge_no_duplicates

def run(*args) :
        
    Acquisition = pd.read_feather(RUN_PATH + "/result_tables/Acquisition.feather")
    Detection = pd.read_feather(RUN_PATH + "/result_tables/Detection.feather")
    Spots = pd.read_feather(RUN_PATH + "/result_tables/Spots.feather")
    Drift = pd.read_feather(RUN_PATH + "/result_tables/Drift.feather")
    Gene_map = pd.read_feather(RUN_PATH + "/result_tables/Gene_map.feather")
    Cell = pd.read_feather(RUN_PATH + "/result_tables/Cell.feather")

    #Post-processing
    Spots = Spots_filtering(
        Spots,
        filter_washout=True,
        segmentation_filter=True
    )
    
    ANALYSIS_MODULES = ['all','distributions' ,'density']
    
    #Analysis
    if "distributions" in args or "all" in args :
        
        from .analysis_parameters import distribution_measures
        
        distributions_analysis(
            Acquisition=Acquisition,
            Detection=Detection,
            Cell=Cell,
            Spots=Spots,
            Gene_map=Gene_map,
            disibutions_measures= distribution_measures
        )
    
    if "density" in args  or "all" in args:
        
        from .analysis_parameters import min_diversity, min_spots_number, cluster_radius
        density_analysis(
            Acquisition=Acquisition,
            Detection=Detection,
            Spots=Spots,
            Gene_map=Gene_map,
            min_number_spots=min_spots_number,
            min_diversity=min_diversity,
            cluster_radius=cluster_radius
        ) 