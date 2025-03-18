"""
Submodule for data post processing, eg Filtering...
"""

import pandas as pd
from .analysis_parameters import FILTER_RNA

def Spots_filtering(
    Spots : pd.DataFrame, 
    filter_washout= True,
    segmentation_filter= True,
    ) :
    
    if filter_washout : 
        Spots = Spots.loc[~Spots['is_washout']]
    
    if segmentation_filter :
        Spots = Spots.loc[Spots['cell_label'] != 0]
    
    return Spots

def RNA_filtering(df_with_target : pd.DataFrame) :
    
    if 'target' not in df_with_target : raise KeyError('"target" column was not found in dataframe columns.')
    
    df_with_target = df_with_target.drop(
        df_with_target[df_with_target['target'].isin(FILTER_RNA)].index,
        axis=0
    )
    
    return df_with_target