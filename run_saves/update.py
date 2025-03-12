import os
import pandas as pd
from Sequential_Fish import __run_cache_path__
from ..tools import get_datetime


def validate_script(RUN_PATH, script) :
    """
    Opens and write in Run_cache that passed script was successfully runned and update datetime.
    """
    
    
    date = get_datetime()
    run_dataframe = pd.read_feather(__run_cache_path__)
    assert script in run_dataframe.columns, "Script was not found in run_dataframe table. Update run_dataframe or check script name matches _pipeline_scripts.py"

    run_dataframe.loc[run_dataframe['RUN_PATH'] == RUN_PATH, [script]] = True
    run_dataframe.loc[run_dataframe['last_modification_date'] == RUN_PATH, [script]] = date
    run_dataframe.reset_index(drop=True).to_feather(__run_cache_path__)

def fail_script(RUN_PATH, script) :
    """
    Opens and write in Run_cache that passed script was successfully runned.
    
    """
    assert script in run_dataframe.columns, "Script was not found in run_dataframe table. Update run_dataframe or check script name matches _pipeline_scripts.py"
    
    run_dataframe = pd.read_feather(__run_cache_path__)
    run_dataframe.loc[run_dataframe['RUN_PATH'] == RUN_PATH, [script]] = True
    run_dataframe.reset_index(drop=True).to_feather(__run_cache_path__)