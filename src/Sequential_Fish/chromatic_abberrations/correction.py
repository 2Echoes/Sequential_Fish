import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from .calibration import calibration_exist, load_calibration
from ..tools import get_voxel_size


def _transform_input(coords: np.ndarray, voxel_size: np.ndarray, poly: PolynomialFeatures) -> np.ndarray:
    """Build polynomial inputs for either a 3D or single-plane calibration."""
    if poly.n_features_in_ == 2:
        return coords[:, 1:] * voxel_size[1:]
    if poly.n_features_in_ == 3:
        return coords * voxel_size
    raise ValueError(f"Unsupported calibration dimensionality: {poly.n_features_in_}")

def apply_polynomial_transform_to_signal(
        image : np.ndarray, 
        poly : PolynomialFeatures, 
        model_x : LinearRegression, 
        model_y : LinearRegression, 
        voxel_size : np.ndarray,
        model_z : LinearRegression | None = None, 
        ):
    """Warp 3D image using learned polynomial transform."""
    z, y, x = image.shape
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    voxel_size = np.asarray(voxel_size)

    X_poly = poly.transform(_transform_input(coords, voxel_size, poly))
    if not model_z is None :
        print("correcting z") 
        new_z_nm = model_z.predict(X_poly)
    else :
        print("not correcting z")
        new_z_nm = coords[:, 0] * voxel_size[0]
    new_y_nm = model_y.predict(X_poly)
    new_x_nm = model_x.predict(X_poly)

    #convert back to pixel
    new_coords_pixel = np.stack([new_z_nm, new_y_nm, new_x_nm], axis=0) / voxel_size[:, None]

    warped = map_coordinates(image, new_coords_pixel, order=1, mode='reflect').reshape(z, y, x)
    return warped

def apply_polynomial_transform_spots(
        coords : np.ndarray,
        poly : PolynomialFeatures,
        model_x : LinearRegression,
        model_y : LinearRegression,
        voxel_size : np.ndarray,
        model_z : LinearRegression = None,
        ) :
    """
    Correct chromatic abberrations for spots using pre-calibrated polynomial interpolation.
    """

    coords = np.asarray(coords)
    voxel_size = np.asarray(voxel_size)
    monosomes = poly.transform(_transform_input(coords, voxel_size, poly))
    new_y_nm = model_y.predict(monosomes)
    new_x_nm = model_x.predict(monosomes)
    if not model_z is None :
        new_z_nm = model_z.predict(monosomes)
    else :
        new_z_nm = coords[:,0] * voxel_size[0]
    new_coords_pixel = np.stack([new_z_nm, new_y_nm, new_x_nm], axis=1) / voxel_size

    return new_coords_pixel

def get_polynomial_features(degree : int) :
    poly = PolynomialFeatures(degree)
    return poly


def correct_Spots_dataframe(
        Detection : pd.DataFrame,
        Spots : pd.DataFrame,
        reference_wavelength :int,       
) :

    wavelength_list =  Detection['wavelength'].unique().tolist()
    for wv in wavelength_list :
        if int(wv) == int(reference_wavelength) : continue
        calibration_available = calibration_exist(
            reference_wavelength=reference_wavelength,
            corrected_wavelength=wv,
        )
        if not calibration_available:
            raise FileNotFoundError(
                "No calibration found for reference wavelength: "
                f"{reference_wavelength}nm and corrected wavelength: {wv}nm. "
                "To configure new calibration use command "
                "'python -m Sequential_Fish calibration'."
            )
    
    for wv in wavelength_list :

        if int(wv) == int(reference_wavelength) : continue

        Detection_loc = Detection.loc[Detection['wavelength'] == wv]
        calibration = load_calibration(reference_wavelength, wv)
        detection_ids_to_correct = Detection_loc['detection_id']
        voxel_size = get_voxel_size(Detection_loc)

        if tuple(voxel_size) != tuple(calibration['voxel_size']) : raise ValueError("Different voxel size for spot detection and calibration : spots {0} ; calibration {1} for reference wavelength {2}nm and corrected wavelength {3}nm.".format(voxel_size, calibration['voxel_size'], reference_wavelength, wv))
        
        spots_idx_to_correct = Spots[Spots['detection_id'].isin(detection_ids_to_correct)].index
        coordinates = np.array(Spots.loc[spots_idx_to_correct, "coordinates"].tolist(), dtype=int)
        new_coordinates = apply_polynomial_transform_spots(
                coords=coordinates,
                poly=calibration['polynomial_features_inv'],
                model_x = calibration['x_inv_fit'],            
                model_y = calibration['y_inv_fit'],
                model_z = calibration['z_inv_fit'],       
                voxel_size=voxel_size 
        ).round().astype(int).tolist()

        Spots.loc[spots_idx_to_correct, ["coordinates"]] = pd.Series(new_coordinates, dtype=object, index= spots_idx_to_correct)

    Spots['z'] = Spots['coordinates'].apply(lambda x: round(x[0])).astype(int)
    Spots['y'] = Spots['coordinates'].apply(lambda x: round(x[1])).astype(int)
    Spots['x'] = Spots['coordinates'].apply(lambda x: round(x[2])).astype(int)

    return Spots