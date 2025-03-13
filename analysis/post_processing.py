"""
Submodule for data post processing, eg Filtering...
"""

import pandas as pd

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