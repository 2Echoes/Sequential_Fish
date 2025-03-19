import os
import pandas as pd
from Sequential_Fish import __run_cache_path__
from ..tools import get_datetime
from .dataframe import add_new_run

from .._pipeline_scripts import PIPELINE_SCRIPTS
from .dataframe import create_run_dataframe


def validate_script(RUN_PATH, script: str) :
    """
    Opens and write in Run_cache that passed script was successfully runned and update datetime.
    """
    
    if script.endswith('.py') : script = script[:-3]
    script = os.path.basename(script)
        
    date = get_datetime()
    run_dataframe = pd.read_feather(__run_cache_path__)

    assert script in run_dataframe.columns, f"Script {script} was not found in run_dataframe table. Update run_dataframe or check script name matches _pipeline_scripts.py"
    
    run_dataframe.loc[run_dataframe['RUN_PATH'] == RUN_PATH, [script]] = True
    run_dataframe.loc[run_dataframe['last_modification_date'] == RUN_PATH, [script]] = date
    run_dataframe.reset_index(drop=True).to_feather(__run_cache_path__)

def fail_script(RUN_PATH, script) :
    """
    Opens and write in Run_cache that passed script was successfully runned.
    
    """
    if script.endswith('.py') : script = script[:-3]
    script = os.path.basename(script)
    run_dataframe = pd.read_feather(__run_cache_path__)
    assert script in run_dataframe.columns, f"Script {script} was not found in run_dataframe table. Update run_dataframe or check script name matches _pipeline_scripts.py"
    
    run_dataframe.loc[run_dataframe['RUN_PATH'] == RUN_PATH, [script]] = True
    run_dataframe.reset_index(drop=True).to_feather(__run_cache_path__)
    
def check_run(run_path) :
    """
    Check if folder is already in cached runs. If not initiate a new run with current parameters and scripts to False.
    """
    run_dataframe = pd.read_feather(__run_cache_path__)
    
    if run_path in list(run_dataframe['RUN_PATH']) :
        print(f"Updating RUN : {run_path}")
    else :
        print(f"Initiating new RUN : {run_path}")
        run_dataframe = add_new_run(run_dataframe)
        run_dataframe.reset_index(drop=True).to_feather(__run_cache_path__)
        
def run_status() :
    run_dataframe = pd.read_feather(__run_cache_path__)
    print(run_dataframe.loc[:,['run_id', 'RUN_PATH',] + PIPELINE_SCRIPTS])
    
