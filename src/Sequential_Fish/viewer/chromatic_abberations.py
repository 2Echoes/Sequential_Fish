from typing import cast


from .types import ThreadedWidget, NapariWidget, UserInputError
from ..tools import get_datetime
from ..chromatic_abberrations import calibration_exist, load_calibration
from ..chromatic_abberrations.calibration import match_beads, fit_polynomial_transform_3d, save_fit_model
from ..chromatic_abberrations import apply_polynomial_transform_spots, apply_polynomial_transform_to_signal


from napari.viewer import Viewer
from napari.types import LayerDataTuple
from napari.layers import Points, Image
from magicgui import magicgui

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class ChromaticWidget(ThreadedWidget) :

    def __init__(self, *,run_path : str, voxel_size : tuple[int,int,int], wavelength_list : list[int], viewer: Viewer):
        self.run_path = run_path
        self.voxel_size =voxel_size
        self.wavelength_list = wavelength_list
        super().__init__(viewer=viewer)


_CHROMATIC_WIDGETS : 'list[NapariWidget]' = []
def register_chromatic_widget(cls) :
    _CHROMATIC_WIDGETS.append(cls)
    return cls

def initiate_chromatic_widgets(
        run_path : str,
        viewer : Viewer,
        wavelength_list : list[int],
        voxel_size : tuple,
) -> tuple[list, list[NapariWidget]]:
    widget_list = []
    linked_widgets = []
    for cls in _CHROMATIC_WIDGETS :
        instance = cls(run_path=run_path, voxel_size=voxel_size, wavelength_list = wavelength_list, viewer=viewer)
        if hasattr(instance,"enabled") :
            if instance.enabled :
                widget_list.extend(instance.get_widgets())
        else :
            widget_list.extend(instance.get_widgets())

        if hasattr(instance, "update") and callable(getattr(instance, "update")):
            linked_widgets.append(instance)

    return widget_list, linked_widgets

@register_chromatic_widget
class SpotCorrector(ChromaticWidget) :
    def _create_widget(self):
        
        @magicgui(
            reference_wavelength = {"choices" : self.wavelength_list},
            layer_wavelenth = {"choices" : self.wavelength_list}
        )
        def correct_spots(
            Spots : Points,
            reference_wavelength : int,
            layer_wavelenth : int
        ) :

            if not calibration_exist(self.run_path, reference_wavelength, corrected_wavelength=layer_wavelenth) :
                raise UserInputError(f"Not calibration was found for reference wavelength : {reference_wavelength}nm and layer wavelength : {layer_wavelenth}")

            calibration = load_calibration(reference_wavelength=reference_wavelength, corrected_wavelength=layer_wavelenth)

            if Spots.data.ndim == 3 :
                spot_array = np.concat([
                    np.zeros(len(Spots.data)).reshape(-1,1),
                    Spots.data,
                    ])
            else :
                spot_array = Spots.data

            new_coordinates = np.concat([
                spot_array[:,0].reshape(-1,1),
                apply_polynomial_transform_spots(
                    coords=spot_array[:,1:], #Selecting spots belonging to one specific fov
                    poly=calibration['polynomial_features_inv'],
                    model_x = calibration['x_inv_fit'],            
                    model_y = calibration['y_inv_fit'],
                ).round().astype(int)
            ],axis=1)

            res = LayerDataTuple((
                new_coordinates,
                {
                    "name" : Spots.name,
                    "size" : Spots.size,
                    "face_color" : Spots.face_color,
                    "opacity" : Spots.opacity,
                    "blending" : Spots.blending,
                    "border_color" : Spots.border_color,
                    "symbol" : Spots.symbol,
                    "scale" : Spots.scale,
                    'units' : "nm",
                    'ndim' : 4,
                },
                'Points'
            ))

            return res

        return correct_spots

@register_chromatic_widget
class SignalCorrector(ChromaticWidget) :
    def _create_widget(self):
        
        @magicgui(
            reference_wavelength = {"choices" : self.wavelength_list},
            layer_wavelenth = {"choices" :self.wavelength_list}
        )
        def correct_spots(
            Signal : Image,
            reference_wavelength : int,
            layer_wavelenth : int
        ) :

            if not calibration_exist(self.run_path, reference_wavelength, corrected_wavelength=layer_wavelenth) :
                raise UserInputError(f"Not calibration was found for reference wavelength : {reference_wavelength}nm and layer wavelength : {layer_wavelenth}")

            calibration = load_calibration(self.run_path,reference_wavelength=reference_wavelength, corrected_wavelength=layer_wavelenth)

            if Signal.data.ndim == 4 :
                new_signal = np.stack(
                    [apply_polynomial_transform_to_signal(
                image=cast(np.ndarray, fov),
                poly=calibration['polynomial_features'],
                model_x = calibration['x_fit'],            
                model_y = calibration['y_fit'],
            ).round().astype(int) for fov in tqdm(cast(np.ndarray, Signal.data), desc="correcting chromatic aberrations", total=len(cast(np.ndarray, Signal.data)))]
                )
            elif Signal.data.ndim == 3 :
                new_signal = apply_polynomial_transform_to_signal(
                    image=cast(np.ndarray, Signal.data),
                    poly=calibration['polynomial_features'],
                    model_x = calibration['x_fit'],            
                    model_y = calibration['y_fit'],
                ).round().astype(int)

            else :
                raise AssertionError("Unforseen dimension")


            res = LayerDataTuple((
                new_signal,
                {
                    "name" : Signal.name,
                    "opacity" : Signal.opacity,
                    "blending" : Signal.blending,
                    "scale" : Signal.scale,
                    "projection_mode" : Signal.projection_mode,
                    "colormap" : Signal.colormap,
                    "contrast_limits" : Signal.contrast_limits,
                    "gamma" : Signal.gamma,
                    'units' : "nm",
                },
                'Image'
            ))

            return res

        return correct_spots

@register_chromatic_widget
class ChromaticAberrationCalibrator(ChromaticWidget) :
    def __init__(self, run_path : str, voxel_size : tuple, viewer : Viewer, wavelength_list):

        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.model_z = LinearRegression()
        self.polynomial_features = PolynomialFeatures()
        self.polynomial_features_inv = PolynomialFeatures()
        self.inv_model_x = LinearRegression()
        self.inv_model_y = LinearRegression()
        self.inv_model_z = LinearRegression()
        self.calibration_folder = CALIBRATION_FOLDER
        self.degree = 2
        self.timestamp = get_datetime()
        self.save_widget = self._create_save_widget()
        
        super().__init__(run_path=run_path, viewer=viewer, voxel_size=voxel_size, wavelength_list=wavelength_list)

        self.register_widget(self.save_widget)

    def _create_widget(self):
        """
        Perform calibration for chromatic abberration correction and create a layer with corrected signal to evaluate quality of fit.
        """

        @magicgui(
                image_abberation={'label' : 'Image to correct :'},
                spatial_reference={'label' : 'Points reference'},
                spatial_reference_shifted={'label' : 'Points with aberrations'},
                location = {"min" : 0},
                degree={'label' : 'Degree'},
                pixel_range = {"min" : 0, "value" : 5},
                auto_call=False,
                call_button= "Correct chromatic aberrations",
        )
        def create_corrected_layer(
            image_abberation : Image,
            spatial_reference : Points,
            spatial_reference_shifted : Points,
            location : int,
            degree : int = self.degree,
            pixel_range : int = 5,
        ) :

            if not tuple(image_abberation.scale) == tuple(spatial_reference.scale) == tuple(spatial_reference_shifted.scale) :
                print(f"Scale is not uniform between selected layers.\nimage to correct : {tuple(image_abberation.scale)}\nreference points : {spatial_reference.scale}\npoints with abberation : {spatial_reference_shifted.scale}")

            coords1 = np.asarray(spatial_reference.data)
            coords2 = np.asarray(spatial_reference_shifted.data)

            if coords1.shape[1] == 4 :
                coords1 = coords1[coords1[:,0] == location]
                coords1 = coords1[:,1:]
            if coords2.shape[1] == 4 :
                coords2 = coords2[coords2[:,0] == location]
                coords2 = coords2[:,1:]
            self.degree = degree

            beads, dist = match_beads(
                coords1= coords1,
                coords2= coords2,
                max_dist= pixel_range
            )

            if "Optical Center" in self.viewer.layers :
                assert hasattr(self.viewer.layers["Optical Center"], "optical_center")
                optical_center = self.viewer.layers["Optical Center"].optical_center
            else :
                optical_center = None

            print("optical center : ", optical_center)

            self.polynomial_features, self.model_x, self.model_y, self.model_z = fit_polynomial_transform_3d(
                                                beads,
                                                dist, 
                                                degree=degree,
                                                center=optical_center
                                                )
            
            self.polynomial_features_inv, self.inv_model_x, self.inv_model_y, self.inv_model_z = fit_polynomial_transform_3d(
                                                dist, 
                                                beads,
                                                degree=degree,
                                                center=optical_center
                                                )
            
            if image_abberation.data.ndim == 4 :
                image_corrected = np.stack(
                    [apply_polynomial_transform_to_signal(
                image=cast(np.ndarray, fov),
                poly=self.polynomial_features,
                model_x=self.model_x,
                model_y=self.model_y,
            ).round().astype(int) for fov in tqdm(cast(np.ndarray, image_abberation.data), desc="correcting chromatic aberrations", total=len(cast(np.ndarray, image_abberation.data)))]
                )
            elif image_abberation.data.ndim == 3 :
                image_corrected = apply_polynomial_transform_to_signal(
                        image=cast(np.ndarray, image_abberation.data),
                        poly=self.polynomial_features,
                        model_x=self.model_x,
                        model_y=self.model_y,
                    ).round().astype(int)
            else :
                raise AssertionError

            res = LayerDataTuple((
                image_corrected,
                {   "name" : "Interpolation result",
                    "opacity" : image_abberation.opacity,
                    "blending" : image_abberation.blending,
                    "scale" : image_abberation.scale,
                    "projection_mode" : image_abberation.projection_mode,
                    "colormap" : image_abberation.colormap,
                    'units' : "nm",
                    "contrast_limits" : image_abberation.contrast_limits,
                    "gamma" : image_abberation.gamma,},
                "Image"

            ))

            return res

        self.timestamp = get_datetime()

        return create_corrected_layer
    
    def _create_save_widget(self) :
        """
        This widget allow user to save previously performed calibration.
        """

        @magicgui(
                auto_call=False, 
                call_button= "Save calibration"
                )
        def save_method(
            reference_wavelength : int,
            corrected_wavelength : int,
        ) :
            
            save_fit_model(
                run_path=self.run_path,
                x_fit=self.model_x,
                y_fit=self.model_y,
                z_fit=self.model_z,
                polynomial_features= self.polynomial_features,
                polynomial_features_inv= self.polynomial_features_inv,
                x_inv_fit=self.inv_model_x,
                y_inv_fit=self.inv_model_y,
                z_inv_fit=self.inv_model_z,
                degree=self.degree,
                timestamp= self.timestamp,
                corrected_wavelength=corrected_wavelength,
                reference_wavelength=reference_wavelength,
            )
        
        return save_method

class OpticalCenterSetter(NapariWidget) :
    def __init__(self, viewer : Viewer, **_):
        super().__init__()
        self.viewer = viewer
        self.point_layer = None
        self.center = None
        self.layer_name = "Optical Center"
        self.listener = None

    def _create_widget(self):

        @magicgui(
                auto_call=False,
                call_button="Set optical center",
                model_points_layer = {'label' : 'Points layer'}
        )
        def create_center_picker(
            model_points_layer : Points
                ) :

            """Create a Points layer that enforces a single point for picking a center."""
            
            if self.layer_name in self.viewer.layers :
                return self.point_layer
            
            center_layer = self.viewer.add_points(
                ndim=model_points_layer.ndim,
                size=20,
                face_color='transparent',
                blending ='additive',
                border_color='gold',
                symbol="cross",
                name=self.layer_name,
                scale = model_points_layer.scale,
                units = model_points_layer.units,
                metadata={'role': 'center_picker'}
            )
            center_layer = cast(Points,center_layer)
            self.point_layer = center_layer
            center_layer.optical_center = self.center

            def _enforce_single_point(event):
                layer : Points  = event.source

                stop_listening()
                if len(layer.data) > 1:
                    # Keep only the most recently added point
                    layer.data = layer.data[-1:]
                    self.center = layer.data[0,-2:] #keep yx coordinates
                    layer.refresh()
                elif len(layer.data) == 0 :
                    self.center = None
                center_layer.optical_center = self.center
                start_listening()

            def start_listening() :
                self.listener = center_layer.events.data.connect(_enforce_single_point)
            def stop_listening() :
                center_layer.events.data.disconnect(self.listener)
                self.listener = None

            start_listening()
            self.viewer.layers.events.connect(self._on_layer_deletion)
            
            return center_layer
        return create_center_picker

    def get_optical_center(self) :
        return self.center

    def _on_layer_deletion(self) :
        if not "Optical Center" in self.viewer.layers :
            self.center = None
