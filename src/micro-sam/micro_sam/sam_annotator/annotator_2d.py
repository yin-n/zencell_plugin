from typing import Optional, Tuple, Union

import napari
import numpy as np

import torch

from .. import util
from . import _widgets as widgets
from ._state import AnnotatorState
from ._annotator import _AnnotatorBase
from .util import _initialize_parser, _sync_embedding_widget
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from qtpy.QtWidgets import QFileDialog
import tifffile
import re
import pandas as pd
import os


class Annotator2d(_AnnotatorBase):
    def _get_widgets(self):
        autosegment = widgets.AutoSegmentWidget(
            self._viewer, with_decoder=AnnotatorState().decoder is not None, volumetric=False
        )
        #----------------------------------------------
        # [CHANGE!] add a saving widget to the annotator
        # "save": save_widget
        #----------------------------------------------
        save_widget = self._create_save_widget()
        return {
            "segment": widgets.segment(),
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear(),
            "save": save_widget,  # add saving widget
        }
    #----------------------------------------------
    # [CHANGE!] create a widget for saving segmentation results
    def _create_save_widget(self):
        """create combine and saving widget"""
       
        widget = QWidget()
        layout = QVBoxLayout()
        
        save_button = QPushButton("Combine and Save!")
        save_button.clicked.connect(self._save_current_segmentation)
        
        layout.addWidget(save_button)
        widget.setLayout(layout)
        
        return widget
    #----------------------------------------------
    # [CHANGE!] saving function for combined segmentation
    def _save_current_segmentation(self):

        """Saving the combined segmentation (committed + MAE_pseudo with offset)"""
        try:
            # get data from commited_objects layer
            committed_layer = None
            for layer in self._viewer.layers:
                if hasattr(layer, 'name') and 'committed_objects' in layer.name:
                    committed_layer = layer
                    break
            
            # get MAE_pseudo layer data
            
            # Corrected pattern
            mae_pattern = re.compile(r'^\d+_MAE_pseudo_\d+_\d+_\d+$')
            mae_pseudo_layer = None
            for layer in self._viewer.layers:
                if hasattr(layer, 'name') and mae_pattern.match(layer.name):
                    mae_pseudo_layer = layer
                    
                    break
            
            # Check whether one of the layers is found
            if committed_layer is None and mae_pseudo_layer is None:
                print("No segmentation layers found to save")
                return
            
            # choose saving file path
            id = int(mae_pseudo_layer.name.split('_')[0])
            file_path, _ = QFileDialog.getSaveFileName(
                None, 
                "Save Combined Segmentation", 
                f"{id}_combined_segmentation.tif", 
                "TIFF files (*.tif *.tiff);;NumPy files (*.npy);;All files (*.*)"
            )
            
            if file_path:
                combined_segmentation = None
                
                if committed_layer is not None:
                    committed_data = committed_layer.data
                    # if dask array, compute it
                    if hasattr(committed_data, 'compute'):
                        committed_data = committed_data.compute()
                    
                    combined_segmentation = committed_data.copy()
                    committed_max = int(committed_data.max())
                    print(f"Committed objects max value: {committed_max}")
                else:
                    # if no committed layer, initialize combined_segmentation
                    if mae_pseudo_layer is not None:
                        mae_data = mae_pseudo_layer.data
                        if hasattr(mae_data, 'compute'):
                            mae_data = mae_data.compute()
                        combined_segmentation = np.zeros_like(mae_data)
                        committed_max = 0
                
                if mae_pseudo_layer is not None:
                    mae_pseudo_data = mae_pseudo_layer.data
                    # Dask array processing
                    if hasattr(mae_pseudo_data, 'compute'):
                        mae_pseudo_data = mae_pseudo_data.compute()
                    
                    # create MAE_pseudo mask
                    mae_mask = mae_pseudo_data > 0
                    
                    # MAE_pseudo add offset
                    mae_offset_data = mae_pseudo_data.copy()
                    mae_offset_data[mae_mask] += committed_max

                    # cut MAE_pseudo_layer data
                    mae_pseudo_layer_shape = mae_pseudo_layer.data.shape
                    z_center = int(mae_pseudo_layer.name.split('_')[-3])
                    y_patch = int(mae_pseudo_layer.name.split('_')[-2])
                    x_patch = int(mae_pseudo_layer.name.split('_')[-1])
                    
                    y_center = mae_pseudo_layer_shape[1] // 2
                    x_center = mae_pseudo_layer_shape[2] // 2
                    mae_offset_data = mae_offset_data[z_center,y_center - y_patch: y_center + y_patch, x_center - x_patch: x_center + x_patch]
                    
                    # make sure combined_segmentation shape is correct
                    if combined_segmentation is None:
                        combined_segmentation = mae_offset_data
                    else:
                        # check whether shapes match
                        if combined_segmentation.shape != mae_offset_data.shape:
                            print(f"Warning: Shape mismatch! Committed: {combined_segmentation.shape}, MAE: {mae_offset_data.shape}")
                            # cut to the same shape
                            combined_segmentation = combined_segmentation[y_center - y_patch: y_center + y_patch, x_center - x_patch: x_center + x_patch]
                        

                        combined_segmentation += mae_offset_data
                    
                    mae_max = int(mae_offset_data.max())
                    print(f"MAE pseudo max value after offset: {mae_max}")
                
                # save the combined segmentation
                if ext := file_path.split('.')[-1] == 'npy':
                    np.save(file_path, combined_segmentation)
                else:
                    try:
                        tifffile.imwrite(file_path, combined_segmentation.astype(np.uint16))
                    except ImportError:
                        from PIL import Image
                        Image.fromarray(combined_segmentation.astype(np.uint16)).save(file_path)
                
                # print some information about the saved segmentation
                unique_values = np.unique(combined_segmentation)
                print(f"Combined segmentation saved to: {file_path}")
                print(f"Combined segmentation shape: {combined_segmentation.shape}")
                print(f"Total unique objects: {len(unique_values) - 1}") 
                print(f"Value range: {unique_values.min()} - {unique_values.max()}")
                # TODO need to save image
                for layer in self._viewer.layers:
                    if hasattr(layer, 'name') and 'FoV - Signal' in layer.name:
                        fov_sig = layer.data
                        fov_sig = fov_sig.compute() if hasattr(fov_sig, 'compute') else fov_sig
                        fov_sig = fov_sig[z_center,y_center - y_patch: y_center + y_patch, x_center - x_patch: x_center + x_patch]
                        break
                
                for layer in self._viewer.layers:
                    if hasattr(layer, 'name') and 'FoV - Reference' in layer.name:
                        fov_ref = layer.data
                        fov_ref = fov_ref.compute() if hasattr(fov_ref, 'compute') else fov_ref
                        fov_ref = fov_ref[z_center,y_center - y_patch: y_center + y_patch, x_center - x_patch: x_center + x_patch]
                        break
                #---------------------------------------------------------------------------
                # save image and information 
                
                img = np.stack([fov_ref, fov_sig], axis=0)  # shape: (2, H, W)
                if file_path.endswith('.npy'):
                    image_tif_path = file_path.replace('.npy', '_image.tif')
                image_tif_path = file_path.replace('.tif', '_image.tif')

                tifffile.imwrite(image_tif_path, img)

                # also need to save the information about the segmentation
              
                cur_df = pd.read_csv("./meta_info/current_points.csv")

               
                folder = os.path.dirname(image_tif_path)
                csv_path = os.path.join(folder, "all_points.csv")

                all_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=cur_df.columns)
                all_csv = pd.concat([all_csv, cur_df], ignore_index=True, sort=False)
                all_csv.to_csv(csv_path, index=False)
        
        except Exception as e:
            print(f"Error saving combined segmentation: {e}")
            import traceback
            traceback.print_exc()

    
    # TODO change here to add default checkpoint path 
    def _create_widgets(self):
        # super parent class _create_widgets
        super()._create_widgets()

        # add embeddings widget weights path
        default_weights_path = "/mnt/aperto/yin/sam_sc_ckpt/latest.pt"
        self._widgets["embeddings"].custom_weights = default_weights_path
        self._widgets["embeddings"].custom_weights_param.setText(default_weights_path)
        # add default device
        self._widgets["embeddings"].device = "cuda"
        self._widgets["embeddings"].device_dropdown.setCurrentText("cuda")
   



    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__(viewer=viewer, ndim=2)


def annotator_2d(
    image: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None, # change here to a default checkpoint path
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Start the 2d annotation tool for a given image.

    Args:
        image: The image data.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 3 else image.shape

    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, precompute_amg_state=precompute_amg_state,
        ndim=2, checkpoint_path=checkpoint_path, device=device, prefer_decoder=prefer_decoder,
        skip_load=False, use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator2d(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
    annotator._update_image(segmentation_result=segmentation_result)

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(description="Run interactive segmentation for an image.")
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    annotator_2d(
        image, embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        precompute_amg_state=args.precompute_amg_state, checkpoint_path=args.checkpoint,
        device=args.device, prefer_decoder=args.prefer_decoder,
    )
