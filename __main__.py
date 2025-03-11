import sys
import os
import pandas as pd
from Sequential_Fish import viewer, pipeline
from Sequential_Fish.pipeline.runner import launch_script, script_folder
from Sequential_Fish.run_saves import create_run_dataframe, check_run_dataframe
from Sequential_Fish._pipeline_scripts import PIPELINE_SCRIPTS

script_dir = os.path.dirname(os.path.abspath(__file__))
run_cache_path = os.path.join(script_dir, "run_saves", "Run_cache.feather")


def main():

    MODULES = ['viewer', 'pipeline']

    #RUN CACHE
    if not os.path.isfile(run_cache_path) :
        print(f"Creating run_cache at {run_cache_path}")
        Runs = create_run_dataframe()
        Runs.reset_index(drop=True).to_feather(run_cache_path)
    else :
        Runs = pd.read_feather(run_cache_path)
        Runs = check_run_dataframe(Runs)
        Runs.reset_index(drop=True).to_feather(run_cache_path)
    
    #CALL TO MODULES
    if len(sys.argv) < 2:
        print("Usage: python -m my_package <module> [args...]")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

    module = sys.argv[1]
    submodules = sys.argv[2:]

    if module == "viewer":

        if len(submodules) > 0 :
            print(f"No argument for viewer. Ignoring passed arguments : {submodules}")

        viewer.run()
        
    elif module == "pipeline":
        if len(submodules) == 0 :
            pipeline.run()
        else :
            if not all([script in PIPELINE_SCRIPTS for script in submodules]) :
                print(f"Unknown pipeline scripts. \nChoose from : {PIPELINE_SCRIPTS}")
            else :
                for script in submodules : 
                    if not script.endswith('.py') : script += ".py"
                    launch_script(script_folder + '/' + script)
    else:
        print(f"Unknown module: {module}")
        print("Available modules: {0}".format(MODULES))
        sys.exit(1)

if __name__ == "__main__":
    main()
