import subprocess
import logging
import os
from datetime import datetime

"""
Main script to launch pipeline

Script are listed in `scripts_rounds` in corresponding order.  
All scripts of a round are launched even if an error is returned within the round but next round will be launched only if all previous scripts were sucessful.

"""
script_folder = os.path.abspath(__file__)
script_folder = os.path.dirname(script_folder)

scripts_rounds = {
    'input' : ['input.py'],
    'analysis' : ['detection.py', 'segmentation.py', 'drift.py'],
    'alignement' : ['alignement.py'],
    'filtering' : ['washout.py'],
    'quantification' : ['quantification.py']
}

from Sequential_Fish.pipeline_parameters import RUN_PATH

log_file = RUN_PATH + "/run_log.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

### script launcher function
def launch_script(script_name):
    """Launch script from pipeline."""
    try:
        logging.info(f"Exécution du script : {script_name}")
        script_start = datetime.now()
        result = subprocess.run(
            ["python", script_name],
            check=True,
            text=True,
            capture_output=True
        )
        script_end = datetime.now()
        run_duration = (script_end - script_start).total_seconds()
        
        logging.info("script duration : {0}".format(run_duration))
        logging.info(f"script succeed {script_name}:\n{result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        script_end = datetime.now()
        run_duration = (script_end - script_start).total_seconds()

        logging.info("script duration : {0}".format(run_duration))
        logging.error(f"script failed {script_name}:\n{e.stderr}")

        return False

def main():
    start_time = datetime.now()
    logging.info("NEW RUN")
    
    for round_key in scripts_rounds.keys() :
        scripts = scripts_rounds[round_key]
        sucess = []
        for script in scripts:
            if os.path.exists(script_folder + '/' + script):
                result = launch_script(script_folder + '/' + script)
                sucess.append(result)
            else:
                logging.warning(f"File {script} not found.")
                sucess.append(False)

        if all(sucess) :
            logging.info("Step {0} succeed.".format(round_key))
        else :
            logging.error("Step {0} failed, ending run.".format(round_key))
            quit()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Run ends. Total duration : {duration:.2f} seconds")

if __name__ == "__main__":
    main()