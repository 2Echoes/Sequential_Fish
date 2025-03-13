import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from .utils import distribution_super_plot
from ..tools import safe_merge_no_duplicates
from ..pipeline_parameters import RUN_PATH

def merge_data(
    Acquisition : pd.DataFrame,
    Detection : pd.DataFrame,
    Cell : pd.DataFrame,
    Spots : pd.DataFrame,
    Gene_map : pd.DataFrame,
) :
    """
    Returns : Detection, Cell, Spots
    """
    Detection = safe_merge_no_duplicates(
        Detection,
        Acquisition,
        on= 'acquisition_id',
        keys= ['cycle']
    )
    
    Detection = safe_merge_no_duplicates(
        Detection,
        Gene_map,
        on= ['cycle', 'color_id'],
        keys= 'target'
    )
    
    Spots = safe_merge_no_duplicates(
        Spots,
        Detection,
        on='detection_id',
        keys= ['target', 'location']
    )
    
    Cell = safe_merge_no_duplicates(
        Cell,
        Detection,
        on='detection_id',
        keys='target'
    )
    
    return Detection, Cell, Spots

def distributions_analysis(
    Acquisition : pd.DataFrame,
    Detection : pd.DataFrame,
    Cell : pd.DataFrame,
    Spots : pd.DataFrame,
    Gene_map : pd.DataFrame,
    disibutions_measures : 'list[str]',
) :
    output_path = RUN_PATH + "/analysis/distribution_analysis/"
    os.makedirs(output_path, exist_ok=True)
    
    log_file = output_path + "/distribution_analysis_log.log"
    logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force= True
)
    
    try :
        print("Starting distribution analysis...")
        logging.info(f"New density analysis")
        logging.info(f"Distribution_measures :\n{disibutions_measures}")
        
        Detection, Cell, Spots = merge_data(
            Acquisition=Acquisition,
            Detection=Detection,
            Cell=Cell,
            Spots=Spots,
            Gene_map=Gene_map
        )
        Cell = Cell.loc[~Cell['target'].str.contains('Washout')]

        for measure in disibutions_measures :
        
            data = Cell.groupby('target')[measure].apply(list)

            fig = plt.figure(figsize=(16,8))
            ax = fig.gca()
            ax = distribution_super_plot(
                data,
                ax,
                ylabel=measure,
                title= f"Distribution of {measure} per cell",
            )

            if 'index' in measure :
                min_x,max_x,min_y,max_y = plt.axis()
                ax.plot([min_x, max_x], [1,1], '--b')

            plt.savefig(output_path + f"/{measure}.svg")
            plt.close()
    
    except Exception as e :
        logging.error(f"analysis failed :\n{e.stderr}")
        
        return False
        
    else :
        logging.info(f"analysis succeed")
        
        return True
        