import os
import dask.array as da
import napari
import numpy as np
import torch
import zarr
from app_model.backends.qt import QMenuItemAction
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget
from PyQt5 import QtGui  
from ome_zarr.io import parse_url      
from ome_zarr.reader import Reader  

from zencell.zencell_model import models_vit
from zencell.zencell_model.cellpose.dynamics import compute_masks
import json
import pandas as pd


class CropQWidget2D(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._viewer_results = None
        self.points_layer = None
        self._last_points = []
        self._segemented_points = []
        self._cropped_points = {}
        self._default_brain_path = '/mnt/aperto/tatz_brain_data/240620_01_MX007-1/fused.zarr'
        self._default_ckpt = "/mnt/aperto/yin/zencell_ckpt/2D/checkpoint-499.pth"
        self._pseudo_label = None

        self.layout = QtWidgets.QVBoxLayout()

        self.data_shape = [12, 6000, 8000]
        self.resolution = ['0','1','2']

        #--------------------------------------------------------------------
        #  Global Information
        self.global_info_group = QtWidgets.QGroupBox("Global information")
        global_info_layout = QtWidgets.QVBoxLayout()

        # 1. Whole brain browse
        self.brain_dir_label = QtWidgets.QLabel("Brain Path:")
        self.brain_dir_input = QtWidgets.QLineEdit(self._default_brain_path)
        self.brain_browse_button = QtWidgets.QPushButton("Browse")
        self.brain_browse_button.clicked.connect(self.browse_brain_dir)

        brain_dir_layout = QtWidgets.QHBoxLayout()
        brain_dir_layout.addWidget(self.brain_dir_input)
        brain_dir_layout.addWidget(self.brain_browse_button)

        global_info_layout.addWidget(self.brain_dir_label)
        global_info_layout.addLayout(brain_dir_layout)

        # 2. get metadata button
        self.meta_button = QtWidgets.QPushButton("Get metadata")
        self.meta_button.clicked.connect(self.get_metadata)

        global_info_layout.addWidget(self.meta_button)

        self.global_info_group.setLayout(global_info_layout)
        self.layout.addWidget(self.global_info_group)
        #--------------------------------------------------------------------

        #--------------------------------------------------------------------
        # Channel and Resolution selection 
        # 3. Reference and signal channels.
        self.channel_select_group = QtWidgets.QGroupBox("Channel and Resolution selection")
        channel_select_layout = QtWidgets.QVBoxLayout()
        self.ref_channel_label = QtWidgets.QLabel(
            "Reference Channel:"
        )
    
        channel_select_layout.addWidget(self.ref_channel_label)
        
        self.ref_channel_combo = QtWidgets.QComboBox()
        self.ref_channel_combo.addItems([str(i + 1) for i in range(self.data_shape[0])])
        channel_select_layout.addWidget(self.ref_channel_combo)
        self.sig_channel_label = QtWidgets.QLabel(
            "Signal Channel:"
        )
    
        channel_select_layout.addWidget(self.sig_channel_label)
        self.sig_channel_combo = QtWidgets.QComboBox()
        self.sig_channel_combo.addItems([str(i + 1) for i in range(self.data_shape[0])])
        channel_select_layout.addWidget(self.sig_channel_combo)


        # 4. Resolution selection.
        self.resolution_label = QtWidgets.QLabel(
            "Resolution:"
        )
    
        channel_select_layout.addWidget(self.resolution_label)
        
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(self.resolution)
        channel_select_layout.addWidget(self.resolution_combo)
        #--------------------------------------------------------------------
        # add input z 
        #---------------------------------------------------------------------
        # self.z_plane_label = QtWidgets.QLabel(
        #     "z-plane:"
        # )
    
        # channel_select_layout.addWidget(self.z_plane_label)
        # self.z_plane_input = QtWidgets.QLineEdit("160")
        # channel_select_layout.addWidget(self.z_plane_input)


        # 5. A button to show whole brain.
        self.show_whole_brain_button = QtWidgets.QPushButton("Show Whole Brain")
        self.show_whole_brain_button.clicked.connect(self.show_whole_brain)
        channel_select_layout.addWidget(self.show_whole_brain_button)

        self.channel_select_group.setLayout(channel_select_layout)
        self.layout.addWidget(self.channel_select_group)


        #--------------------------------------------------------------------
        
        #--------------------------------------------------------------------

        # prediction part
        self.predict_group = QtWidgets.QGroupBox("Prediction settings")
        predict_layout = QtWidgets.QVBoxLayout()

        #--------------------------------------------------------------------
        # add parts to load pseudo segmentation
        #--------------------------------------------------------------------
        self.pred_dir_label = QtWidgets.QLabel("Prediction input file:")
        self.pred_dir_input = QtWidgets.QLineEdit()
        self.browse_pred_button = QtWidgets.QPushButton("Browse")
        self.browse_pred_button.clicked.connect(self.browse_pred_dir)

        self.load_pred_button = QtWidgets.QPushButton("Load!")
        self.load_pred_button.clicked.connect(self.load_pred_dir)

        pred_dir_layout = QtWidgets.QHBoxLayout()
        pred_dir_layout.addWidget(self.pred_dir_input)

        pred_dir_layout.addWidget(self.browse_pred_button)
        pred_dir_layout.addWidget(self.load_pred_button)

        predict_layout.addWidget(self.pred_dir_label)
        predict_layout.addLayout(pred_dir_layout)



        # 6. input segment shape
        seg_shape_label = QtWidgets.QLabel("Segmentation shape (y, x):")
        predict_layout.addWidget(seg_shape_label)
        self.seg_shape_y_label = QtWidgets.QLabel("y:")
        self.seg_shape_y_input = QtWidgets.QLineEdit("256")
        self.seg_shape_x_label = QtWidgets.QLabel("x:")
        self.seg_shape_x_input = QtWidgets.QLineEdit("256")

        seg_shape_layout = QtWidgets.QHBoxLayout()
        seg_shape_layout.addWidget(self.seg_shape_y_label)  
        seg_shape_layout.addWidget(self.seg_shape_y_input)
        seg_shape_layout.addWidget(self.seg_shape_x_label)
        seg_shape_layout.addWidget(self.seg_shape_x_input)
        predict_layout.addLayout(seg_shape_layout)


        # 7.input image visualize shape:
        vis_shape_label = QtWidgets.QLabel("Image visualization shape (z, y, x):")
        predict_layout.addWidget(vis_shape_label)
        self.vis_shape_z_label = QtWidgets.QLabel("z:")
        self.vis_shape_z_input = QtWidgets.QLineEdit("40")
        self.vis_shape_y_label = QtWidgets.QLabel("y:")
        self.vis_shape_y_input = QtWidgets.QLineEdit("768")
        self.vis_shape_x_label = QtWidgets.QLabel("x:")
        self.vis_shape_x_input = QtWidgets.QLineEdit("768")

        vis_shape_layout = QtWidgets.QHBoxLayout()
        vis_shape_layout.addWidget(self.vis_shape_z_label)  
        vis_shape_layout.addWidget(self.vis_shape_z_input)
        vis_shape_layout.addWidget(self.vis_shape_y_label)  
        vis_shape_layout.addWidget(self.vis_shape_y_input)
        vis_shape_layout.addWidget(self.vis_shape_x_label)
        vis_shape_layout.addWidget(self.vis_shape_x_input)
        predict_layout.addLayout(vis_shape_layout)


        # add a button to show the cropped area
        self.show_cropped_area_button = QtWidgets.QPushButton("Show Cropped Area")
        self.show_cropped_area_button.clicked.connect(self.show_cropped_area_by_points)
        predict_layout.addWidget(self.show_cropped_area_button)

        # 11. clear all points button
        # self.clear_button = QtWidgets.QPushButton("Clear all points")
        # self.clear_button.clicked.connect(self.clear_all_points)
        # predict_layout.addWidget(self.clear_button)

        self.predict_group.setLayout(predict_layout)
        self.layout.addWidget(self.predict_group)


        #-------------------------------------------
        #--------------------------------------------------------------------
        
        # 12. add log at bottom

        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)
        self.setLayout(self.layout)

    def clear_all_points(self):
        """Clear all points in the points layer."""
        if self.points_layer is not None:
            self.points_layer.data = np.empty((0, 3))
            self.log_output.append("All points cleared.")
            self._last_points = []
            self._segemented_points = []
    
    def get_metadata(self): 
        zarr_path = self.brain_dir_input.text()
        
        try:
            zarr_file = zarr.open(zarr_path, mode='r')
            self.resolution = list(zarr_file.keys())
            za_wh = zarr_file[self.resolution[-1]]
            dask_wh = da.from_zarr(za_wh)
            self.data_shape = dask_wh.shape

            #read metadata info
            loc = parse_url(zarr_path, mode="r")        
            reader = Reader(loc)
            nodes = list(reader())                 
            img_node = nodes[0]                     
            pyramid = img_node.data             
            
            # TODO add data size for each resolution
            n = len(self.resolution)
            for i in range(n):
                arr = pyramid[i]
                n_elements = arr.size 
                bytes_per_element = arr.dtype.itemsize  # uint16 is 2 bytes

                # Total Bytes
                total_bytes = n_elements * bytes_per_element

                # To TiB
                total_tib = total_bytes / (1024**4)

                # print(f"Total Bytes: {total_bytes}")
                # print(f"Total TiB: {total_tib:.6f} TiB")
                self.resolution[i] = f"{self.resolution[i]} : {arr.shape}, {total_tib:.6f} TiB"  
          



            meta_dict = img_node.metadata
            self.log_output.append(
                f'<b><span style="color:green;">Metadata info:</span></b>'
            )
            self.log_output.append("==========================")
            self.log_output.append(f'<span style="color:green;">Brain path</span>: {zarr_path}')

            if 'axes' in meta_dict:
                 for i in range(len(meta_dict['axes'])):
                     self.log_output.append(
                      f'<span style="color:green;"> axis {i}</span>: {meta_dict["axes"][i]["name"]},\
                      type: {meta_dict["axes"][i]["type"]}'
                )

            self.log_output.append(f'<span style="color:yellow;">Resolution(shape, data size, voxel scale)</span>')
            if 'coordinateTransformations' in meta_dict:
                for i in range(len(meta_dict['coordinateTransformations'])):
                    self.log_output.append(f"{self.resolution[i]},{meta_dict["coordinateTransformations"][i][0]['scale']}")
            self.log_output.append("==========================")

        except Exception as e:
            self.log_output.append(f"<span style='color:red'>reading metadata exception: {e}</span>")
            return

   


        # update the resolution combobox
        self.resolution_combo.clear()
        self.resolution_combo.addItems(self.resolution)
        # self.log_output.append("update resolution combobox by automatically read metadata")
        # #  update the shape of the data
        # self.log_output.append(f"update shape of the data by automatically read metadata: {self.data_shape}")

        #  update comboboxes for reference and signal channels
        num_channels = self.data_shape[0]

        self.ref_channel_combo.clear()
        self.ref_channel_combo.addItems([str(i) for i in range(num_channels)])

        self.sig_channel_combo.clear()
        self.sig_channel_combo.addItems([str(i) for i in range(num_channels)])
        #self.log_output.append("update selected ref/sig channels by automatically read metadata")
        self.log_output.moveCursor(QtGui.QTextCursor.End) 


    def show_whole_brain(self):
        # --- Gather inputs from UI ---
        zarr_path = self.brain_dir_input.text()
        try:
            ref_channel_value = self.ref_channel_combo.currentText()
            sig_channel_value = self.sig_channel_combo.currentText()
            resolution = self.resolution_combo.currentText().split()[0]  # Get the first part before the space

            #z_plane = int(self.z_plane_input.text().strip())

            ref_channel = int(ref_channel_value)
            sig_channel = int(sig_channel_value)
            

            print("resolution is:", resolution)

            print("reference channel is:", ref_channel)
            print("signal channel is:", sig_channel)


        except Exception:
            print(
                "Error getting channels."
            )
            return

        """Show the whole brain image in a new viewer."""
        sig_chan_dask = da.from_zarr(zarr.open(zarr_path, mode="r")[resolution])
        sig_chan_whole = sig_chan_dask[sig_channel][:]


        ref_chan_dask = da.from_zarr(zarr.open(zarr_path, mode="r")[resolution])
        ref_chan_whole = ref_chan_dask[ref_channel][:]

        # sig_chan_dask = da.from_zarr(zarr.open(zarr_path, mode="r")[resolution])
        # sig_chan_whole = zarr.open(zarr_path, mode="r")[resolution][sig_channel][z_plane-20:z_plane+20]
        # ref_channel_whole = zarr.open(zarr_path, mode="r")[resolution][ref_channel][z_plane-20:z_plane+20]
        self.log_output.append(f"show whole brain image in resolution {resolution} with reference channel {ref_channel} and signal channel {sig_channel}")
        self._viewer.add_image(ref_chan_whole, name="reference channel whole brain", colormap="blue", blending="additive", contrast_limits=[0, 65536])
        self._viewer.add_image(sig_chan_whole, name="signal channel whole brain", colormap="green", blending="additive", contrast_limits=[0, 65536])

        """Add a points layer to the viewer for selecting the center of the segmentation area."""
        # Clear any existing points layer
        if self.points_layer is not None:
            self._viewer.layers.remove(self.points_layer)
            self.log_output.append("remove existing points layer")

        self.log_output.append("add new points layer")
        # Create a new points layer
        self.points_layer = self.points_layer = self._viewer.add_points(
            np.empty((0, 3)),  # set as 3D points
            name='click area center for segmentation',
            size=50,
            opacity=1.0,
        )
        
        self.log_output.append("add points to click area center for segmentation")

        # add listner to points layer
        self.points_layer.events.data.connect(self._on_points_changed)

    def _on_points_changed(self, event):
        points = self.points_layer.data

        # get new points in the layer
        new_points = [tuple(p) for p in points]

        # first time, initialize _last_points
        # if not hasattr(self, '_last_points'):
        #     self._last_points = {}

        old_set = set(self._last_points)
        new_set = set(new_points)

        added = new_set - old_set
        removed = old_set - new_set

        if added and not removed:
            for p in added:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"add point: ({coords_str})")

        if removed and not added:
            for p in removed:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"remove point: ({coords_str})")
        
        if added and removed:
            for p in removed:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"move point from: ({coords_str})")
            for p in added:
                coords_str = ", ".join([f"{x:.2f}" for x in p])
                self.log_output.append(f"move point to: ({coords_str})")

        # update last points
        self._last_points = new_points
       
        # scroll to the end
        self.log_output.moveCursor(QtGui.QTextCursor.End)

    def browse_brain_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.brain_dir_input.setText(dir_path)

    def browse_model_file(self):
        """Open a file selection dialog and update the model file field."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",                    
            "",                                      
            "Model Files (*.pth *.pt *.onnx);;All Files (*)"  
        )
        if file_path:
            self.model_ckpt.setText(file_path)

    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def browse_pred_dir(self):
        """Open a file selection dialog and update the pseudo prediction file field."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",                    
            "",                                      
            "Model Files (*.npy *.tiff);;All Files (*)"  
        )
        if file_path:
            self.pred_dir_input.setText(file_path)

    def load_pred_dir(self):
        """Load the pseudo prediction file and add it to the viewer."""
        pred_path = self.pred_dir_input.text()
        if not os.path.exists(pred_path):
            self.log_output.append(f"<span style='color:red'>Prediction file {pred_path} does not exist.</span>")
            return

        try:
            # Load the pseudo prediction file
            self._pseudo_label = np.load(pred_path)
            self.log_output.append(f"Loaded pseudo prediction from {pred_path}")
            
            # Add the pseudo label to the viewer
            self._viewer.add_labels(self._pseudo_label, name="Pseudo Label", blending="additive")
        except Exception as e:
            self.log_output.append(f"<span style='color:red'>Error loading pseudo prediction: {e}</span>")

    def show_cropped_area(self):  
        # --- Gather inputs from UI ---
        points = self.points_layer.data
        

        # get new points in the layer
        cur_points = [tuple(p) for p in points]

        old_set = set(self._segemented_points)
        new_set = set(cur_points)
        added = new_set - old_set

        resolution = int(self.resolution_combo.currentText().split()[0])

        # segment area
        
        seg_y_input = int(self.seg_shape_y_input.text().strip())
        seg_x_input = int(self.seg_shape_x_input.text().strip())
        
        seg_patch_y = seg_y_input// (2*2**resolution)
        seg_patch_x = seg_x_input// (2*2**resolution)


        # visualization area
        vis_z_input = int(self.vis_shape_z_input.text().strip())
        vis_y_input = int(self.vis_shape_y_input.text().strip())
        vis_x_input = int(self.vis_shape_x_input.text().strip())
        patch_z = vis_z_input// (2*2**resolution)
        patch_y = vis_y_input// (2*2**resolution)
        patch_x = vis_x_input// (2*2**resolution)


        boxes_2d = []

        boxes_2d_seg = []

        for point in added:
            z0, y0, x0 = [i for i in point]
            zmin = int(z0 - patch_z)
            zmax = int(z0 + patch_z)
            ymin = int(y0 - patch_y)
            ymax = int(y0 + patch_y)
            xmin = int(x0 - patch_x)
            xmax = int(x0 + patch_x)

            ymin_seg = int(y0 - seg_patch_y)
            ymax_seg = int(y0 + seg_patch_y)
            xmin_seg = int(x0 - seg_patch_x)
            xmax_seg = int(x0 + seg_patch_x)


            # add a rectangle for the cropped area
            rect_seg = [
                [z0, ymin_seg, xmin_seg],  # top-left (z, y, x)
                [z0, ymin_seg, xmax_seg],  # top-right
                [z0, ymax_seg, xmax_seg],  # bottom-right
                [z0, ymax_seg, xmin_seg],  # bottom-left
            ]
            boxes_2d_seg.append(rect_seg)

        
            for z in range(zmin, zmax + 1):
                rect = [
                    [z, ymin, xmin],  # top-left (z, y, x)
                    [z, ymin, xmax],  # top-right
                    [z, ymax, xmax],  # bottom-right
                    [z, ymax, xmin],  # bottom-left
                ]
                boxes_2d.append(rect)

        self._viewer.add_shapes(
            boxes_2d,
            shape_type='rectangle',
            edge_color='blue',
            face_color='transparent',
            edge_width=1,
            name='visualization area'
        )

        self._viewer.add_shapes(
            boxes_2d_seg,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=1,
            name='segment area'
        )

    def show_cropped_area_by_points(self):  
        # --- Gather inputs from UI ---
        points = self.points_layer.data
        selected_indices = set(self.points_layer.selected_data)

        for idx in selected_indices:
            select_point = self.points_layer.data[idx]

            print("!!!selected indices:", idx, select_point)

        

        # get new points in the layer
        cur_points = select_point

        # old_set = set(self._segemented_points)
        # new_set = set(cur_points)
        # added = new_set - old_set

        resolution = int(self.resolution_combo.currentText().split()[0])

        # segment area
        
        seg_y_input = int(self.seg_shape_y_input.text().strip())
        seg_x_input = int(self.seg_shape_x_input.text().strip())
        
        seg_patch_y = seg_y_input// (2*2**resolution)
        seg_patch_x = seg_x_input// (2*2**resolution)


        # visualization area
        vis_z_input = int(self.vis_shape_z_input.text().strip())
        vis_y_input = int(self.vis_shape_y_input.text().strip())
        vis_x_input = int(self.vis_shape_x_input.text().strip())
        patch_z = vis_z_input// (2*2**resolution)
        patch_y = vis_y_input// (2*2**resolution)
        patch_x = vis_x_input// (2*2**resolution)


        boxes_2d = []
        boxes_2d_seg = []

        for idx in selected_indices:
            point = self.points_layer.data[idx]
           
            z0, y0, x0 = [int(i) for i in point]
            
            zmin = int(z0 - patch_z)
            zmax = int(z0 + patch_z)
            ymin = int(y0 - patch_y)
            ymax = int(y0 + patch_y)
            xmin = int(x0 - patch_x)
            xmax = int(x0 + patch_x)

            ymin_seg = int(y0 - seg_patch_y)
            ymax_seg = int(y0 + seg_patch_y)
            xmin_seg = int(x0 - seg_patch_x)
            xmax_seg = int(x0 + seg_patch_x)


            # add a rectangle for the cropped area
            rect_seg = [
                [z0, ymin_seg, xmin_seg],  # top-left (z, y, x)
                [z0, ymin_seg, xmax_seg],  # top-right
                [z0, ymax_seg, xmax_seg],  # bottom-right
                [z0, ymax_seg, xmin_seg],  # bottom-left
            ]
            boxes_2d_seg.append(rect_seg)

        
            for z in range(zmin, zmax + 1):
                rect = [
                    [z, ymin, xmin],  # top-left (z, y, x)
                    [z, ymin, xmax],  # top-right
                    [z, ymax, xmax],  # bottom-right
                    [z, ymax, xmin],  # bottom-left
                ]
                boxes_2d.append(rect)

            # save points and segment area and vis area information in a dictionary
            self._cropped_points[idx] = {
                'id': idx,
                'point': [z0, y0, x0],
                "vis_z": vis_z_input,
                "vis_y": vis_y_input,
                "vis_x": vis_x_input,
                "seg_y": seg_y_input,
                "seg_x": seg_x_input,
                'ref_chn': int(self.ref_channel_combo.currentText()),
                'sig_chn': int(self.sig_channel_combo.currentText()),
                'brain_path': self.brain_dir_input.text(),

            }

            print(f"add cropped area for point {idx}: {self._cropped_points[idx]}")
        

        print('all points in cropped is:', self._cropped_points)

        self._viewer.add_shapes(
            boxes_2d,
            shape_type='rectangle',
            edge_color='blue',
            face_color='transparent',
            edge_width=1,
            name='visualization area'
        )

        self._viewer.add_shapes(
            boxes_2d_seg,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=1,
            name='segment area'
        )


        
        self._viewer_cropped = napari.Viewer()
        self._viewer_cropped.window._qt_window.setWindowTitle("2D Cropped Results")
        
        # crop original image through visualization box

        zarr_path = self.brain_dir_input.text()

        ref_channel_value = self.ref_channel_combo.currentText()
        sig_channel_value = self.sig_channel_combo.currentText()
        resolution = int(self.resolution_combo.currentText().split()[0])

        ref_chn = int(ref_channel_value)
        sig_chn = int(sig_channel_value)

        zarr_file= da.from_zarr(zarr.open(zarr_path, mode="r")['0'])
        img_ref = zarr_file[ref_chn][zmin : zmax, ymin:ymax, xmin:xmax].compute()
        img_sig = zarr_file[sig_chn][zmin : zmax, ymin:ymax, xmin:xmax].compute()

        self._viewer_cropped.add_image(img_ref, name="FoV - Reference",colormap = "magenta", blending="additive")
        self._viewer_cropped.add_image(img_sig, name="FoV - Signal", colormap="green", blending="additive")
        
        # crop segment through segment box
        seg_cropped = self._pseudo_label[ymin_seg:ymax_seg, xmin_seg:xmax_seg]
        seg_full = np.zeros_like(img_ref)
        z_center = img_ref.shape[0] // 2  # Assuming z is the first dimension
        seg_full[z_center, patch_y - seg_patch_y:patch_y + seg_patch_y , patch_x - seg_patch_x:patch_x + seg_patch_x] = seg_cropped
        if hasattr(seg_full, "compute"):
            seg_full = seg_full.compute()
        self._viewer_cropped.add_labels(seg_full, name=f"{idx}_MAE_pseudo_{z_center}_{seg_patch_y}_{seg_patch_x}", blending="additive")

        segmen_2d = img_sig[z_center,patch_y - seg_patch_y:patch_y + seg_patch_y, patch_x - seg_patch_x:patch_x + seg_patch_x]
        # segmen_2d = np.zeros_like(img_sig[z_center])
        # segmen_2d[patch_y - seg_patch_y:patch_y + seg_patch_y, patch_x - seg_patch_x:patch_x + seg_patch_x] = img_sig[z_center,patch_y - seg_patch_y:patch_y + seg_patch_y, patch_x - seg_patch_x:patch_x + seg_patch_x]
        # segmen_2d = segmen_2d.compute() if hasattr(segmen_2d, "compute") else segmen_2d
        #self._viewer_cropped.add_image(segmen_2d, name="Segment - Plane", colormap="green", blending="additive")


        offset_y = patch_y - seg_patch_y
        offset_x = patch_x - seg_patch_x

        # seg_layer = self._viewer_cropped.add_image(
        #     segmen_2d,
        #     name="Segment - Plane",
        #     colormap="green",
        #     blending="additive",
        #     translate=(offset_y, offset_x)
        # )
        # seg_layer = self._viewer_cropped.add_image(
        #     segmen_2d[np.newaxis],  # 加上 np.newaxis 让它变成 (1, H, W)，也就是一个单层的 3D 图像
        #     name="Segment - Plane",
        #     colormap="green",
        #     blending="additive",
        #     translate=(z_center, offset_y, offset_x),  # z_value 是你希望显示在的 z 层
        # )

        center_z = z_center
        center_y = offset_y + segmen_2d.shape[0] / 2
        center_x = offset_x + segmen_2d.shape[1] / 2

        self._viewer_cropped.camera.center = (center_z, center_y, center_x)

        self._viewer_cropped.dims.current_step = (center_z, 0, 0) 
        print(f"Added Segment - Plane at YX offset: {(offset_y, offset_x)}")
        print(f"Set viewer camera center to: {(center_z, center_y, center_x)}")
        print(f"Set current Z slice to: {center_z}")

        # add a rectangle for the cropped segment area
        z = z_center  
        y1 = patch_y - seg_patch_y
        y2 = patch_y + seg_patch_y
        x1 = patch_x - seg_patch_x
        x2 = patch_x + seg_patch_x

        
        rectangle = np.array([
            [z, y1, x1],
            [z, y1, x2],
            [z, y2, x2],
            [z, y2, x1],
        ])

       
        self._viewer_cropped.add_shapes(
            [rectangle],
            shape_type='polygon',
            edge_color='red',
            face_color='transparent',
            name='seg_box',
        )

        self.log_output.append(f"segment cropped area shape: {seg_cropped.shape}")
        # segment current points information
      
        

        data = []
        item_copy = self._cropped_points[idx].copy()
        data.append(item_copy)
        df = pd.DataFrame(data)

        if not os.path.exists('./meta_info'):
            os.makedirs('./meta_info')

        df.to_csv('./meta_info/current_points.csv', index=False)


        def _on_click(self):
            print("napari has", len(self.viewer.layers), "layers")
