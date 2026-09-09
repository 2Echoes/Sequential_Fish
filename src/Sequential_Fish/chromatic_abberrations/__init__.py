"""
Module to handle chromatic abberrations corrections and its calibration.
"""

from .calibration import load_calibration
from .calibration import calibration_exist
from .calibration import get_calibration_folder

from .correction import apply_polynomial_transform_spots
from .correction import apply_polynomial_transform_to_signal
from .correction import get_polynomial_features
from .correction import correct_Spots_dataframe

from .constant import CALIBRATION_FOLDER

<<<<<<< HEAD
def run(run_path, *_) :
    from .launch_calibration import main
    main(run_path)
=======
def run(run_path, *args) :
    raise NotImplementedError("This script was moved to the viewer module")
>>>>>>> a2061ba6e64211497e0bb40be5dd73c087d7592d
