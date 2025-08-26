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

import pandas as pd

from zencell.zencell_model import models_vit
from zencell.zencell_model.cellpose.dynamics import compute_masks


class InferQWidget2D(QWidget):
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

        # 8. Output directory with a browse button.

      
        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QtWidgets.QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)
        predict_layout.addWidget(self.output_dir_label)
        predict_layout.addLayout(output_dir_layout)

        # 9. Model selection also change to use browse button
        self.model_label = QtWidgets.QLabel("Model Path (.pth):")
        self.model_ckpt = QtWidgets.QLineEdit(self._default_ckpt)
        self.model_browse_button = QtWidgets.QPushButton("Browse")
        self.model_browse_button.clicked.connect(self.browse_model_file)
        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(self.model_ckpt)
        model_layout.addWidget(self.model_browse_button)
        predict_layout.addWidget(self.model_label)
        predict_layout.addLayout(model_layout)


        # 10. run Inference button.

        self.run_button = QtWidgets.QPushButton("Predict!")
        self.run_button.clicked.connect(self.run_inference_test)
        predict_layout.addWidget(self.run_button)

        # 11. clear all points button
        self.clear_button = QtWidgets.QPushButton("Clear all points")
        self.clear_button.clicked.connect(self.clear_all_points)
        predict_layout.addWidget(self.clear_button)

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
        sig_chan_whole = zarr.open(zarr_path, mode="r")[resolution][sig_channel][:]
        ref_channel_whole = zarr.open(zarr_path, mode="r")[resolution][ref_channel][:]
        self.log_output.append(f"show whole brain image in resolution {resolution} with reference channel {ref_channel} and signal channel {sig_channel}")
        self._viewer.add_image(ref_channel_whole, name="reference channel whole brain", colormap="blue", blending="additive")
        self._viewer.add_image(sig_chan_whole, name="signal channel whole brain", colormap="green", blending="additive")

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
            size=5,
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


    def run_inference_test(self):
        """Collect UI parameters and execute the backend inference routine."""
        # --- Gather inputs from UI ---

        if not self._cropped_points:
            self.log_output.append("No points added and areas cropped. Please add new points and crop in the viewer.")
            return
        

        zarr_path = self.brain_dir_input.text()
        try:
            ref_channel_value = self.ref_channel_combo.currentText()
            sig_channel_value = self.sig_channel_combo.currentText()
            resolution = int(self.resolution_combo.currentText().split()[0])

            ref_chn = int(ref_channel_value)
            sig_chn = int(sig_channel_value)
            
            print("reference channel is:", ref_chn)
            print("signal channel is:", sig_chn)
            print("resolution is:", resolution)

        except Exception:
            print(
                "Error getting channels."
            )
            return
       


        output_dir = self.output_dir_input.text()

        # --- Set backend parameters ---
        # For this example, the checkpoint path is hardcoded.


        ckpt = self.model_ckpt.text()
        model_name = "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep10_roi256_ctx1280_layerscale"

        # For simplicity, we assume a single process execution.
        rank = 0
        world_size = 1
        torch.cuda.set_device(rank % torch.cuda.device_count())

        # Load the model from the models_vit module.
        try:
            model = models_vit.__dict__[model_name](out_chans=3)
        except Exception as e:
            print("Error loading model:", e)
            return
        model.half().eval().cuda()
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        )



        for idx in self._cropped_points:


            point_info = self._cropped_points[idx]
            point = point_info['point']
            vis_z = point_info['vis_z']
            vis_y = point_info['vis_y']
            vis_x = point_info['vis_x']
            seg_y_input = point_info['seg_y']
            seg_x_input = point_info['seg_x']

                    # segment shape handle if not the same size as model input
            seg_z = 10

            if seg_y_input % 256 == 0:
                seg_y = seg_y_input
                
            else:
                seg_y = seg_y_input//256 * 256 + 256
                    

            if seg_x_input % 256 == 0:
                seg_x = seg_x_input

            else:
                seg_x = seg_x_input//256 * 256 + 256


            print(f"Point {idx}: {point}, vis_z: {vis_z}, vis_y: {vis_y}, vis_x: {vis_x}, seg_y: {seg_y}, seg_x: {seg_x}")
            
            
            z0, y0, x0 = point

            z0, y0, x0 = [int((2**resolution)*i) for i in point]
            print(f"current segment: z={z0}, y={y0}, x={x0}....")
            # make z0 as center of the patch
            z0 -= 4
            y0 -= seg_y // 2
            x0 -= seg_x // 2
           
            zarr_file = zarr.open(zarr_path, mode='r')
            self.resolution = list(zarr_file.keys())
            za_large = zarr_file['0']
            dask_large = da.from_zarr(za_large)

            # Get the shape of the whole brain image. if it has channel in first dimension
            if len(dask_large.shape) == 4:
                zmax, ymax, xmax = dask_large.shape[1:]
            else:
                zmax, ymax, xmax = dask_large.shape
            self.log_output.append(f"whole brain shape with highest resolution is: {dask_large.shape}")

            print(f"Volume dimensions: zmax={zmax}, ymax={ymax}, xmax={xmax}")

            # Use the segmentation shape provided by the user.
            z, y, x = seg_z, seg_y, seg_x

            # --- Compute model-specific patch sizes and paddings ---
            rz, ry, rx = model.patch_embed_roi.img_size
            cz, cy, cx = model.patch_embed_ctx.img_size
            az, ay, ax = model.patch_embed_ctx.patch_size

            print(f"Model ROI patch size (rz, ry, rx): ({rz}, {ry}, {rx})")
            print(f"Segmentation shape (z, y, x): ({z}, {y}, {x})")

            # Ensure that the segmentation shape is divisible by the ROI patch size.
            try:
                assert (
                    z % rz == 0 and y % ry == 0 and x % rx == 0
                ), "Segmentation shape must be divisible by the ROI patch size."
            except AssertionError as e:
                print(e)
                return

            zpad_head = (cz - rz) // 2 // az * az
            ypad_head = (cy - ry) // 2 // ay * ay
            xpad_head = (cx - rx) // 2 // ax * ax
            zpad_tail = cz - rz - zpad_head
            ypad_tail = cy - ry - ypad_head
            xpad_tail = cx - rx - xpad_head

            def crop_with_pad(chn, z0, y0, x0):
                """Crop the desired volume with padding if necessary."""
                z_st = z0 - zpad_head - rz // 2
                z_ed = z0 + z + zpad_tail + rz // 2
                y_st = y0 - ypad_head - ry // 2
                y_ed = y0 + y + ypad_tail + ry // 2
                x_st = x0 - xpad_head - rx // 2
                x_ed = x0 + x + xpad_tail + rx // 2
                z_slice = slice(max(z_st, 0), min(z_ed, zmax))
                y_slice = slice(max(y_st, 0), min(y_ed, ymax))
                x_slice = slice(max(x_st, 0), min(x_ed, xmax))
                z_pad = (max(-z_st, 0), max(z_ed - zmax, 0))
                y_pad = (max(-y_st, 0), max(y_ed - ymax, 0))
                x_pad = (max(-x_st, 0), max(x_ed - xmax, 0))
                arr = zarr.open(zarr_path, mode="r")["0"][
                    chn, z_slice, y_slice, x_slice
                ]
                if any(a > 0 or b > 0 for a, b in (z_pad, y_pad, x_pad)):
                    arr = np.pad(arr, (z_pad, y_pad, x_pad))
                return arr

            # Crop the region for both channels.
            ref_arr = crop_with_pad(ref_chn, z0, y0, x0)
            sig_arr = crop_with_pad(sig_chn, z0, y0, x0)

            # --- Set up sliding window offsets for segmentation ---
            z_offs_list, y_offs_list, x_offs_list = np.meshgrid(
                np.arange(0, z + 1, rz // 2),
                np.arange(0, y + 1, ry // 2),
                np.arange(0, x + 1, rx // 2),
            )
            z_offs_list = z_offs_list.reshape(-1).tolist()
            y_offs_list = y_offs_list.reshape(-1).tolist()
            x_offs_list = x_offs_list.reshape(-1).tolist()

            # Prepare tensors for accumulating the outputs.
            cell_prob = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
            cell_flow = torch.zeros([2, z + rz, y + ry, x + rx], device="cuda")
            weight = torch.zeros([z + rz, y + ry, x + rx], device="cuda")

            # Create a weight template for blending the patches.
            z_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rz) - 1.0) ** 2
            y_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=ry) - 1.0) ** 2
            x_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rx) - 1.0) ** 2
            weight_template = (
                3.0
                - z_dist_sq.view(-1, 1, 1)
                - y_dist_sq.view(1, -1, 1)
                - x_dist_sq.view(1, 1, -1)
            ) / 3.0
            weight_template = weight_template.cuda()

            # --- Run inference over the sliding window ---
            with torch.no_grad():
                ref_arr = torch.from_numpy(ref_arr).cuda()
                sig_arr = torch.from_numpy(sig_arr).cuda()
                for z_offs, y_offs, x_offs in zip(
                    z_offs_list, y_offs_list, x_offs_list, strict=False
                ):
                    ref_slice = ref_arr[
                        z_offs : z_offs + cz,
                        y_offs : y_offs + cy,
                        x_offs : x_offs + cx,
                    ]
                    sig_slice = sig_arr[
                        z_offs : z_offs + cz,
                        y_offs : y_offs + cy,
                        x_offs : x_offs + cx,
                    ]
                    input_vol = torch.stack([ref_slice, sig_slice], dim=0).float()
                    # Normalize input.
                    input_vol = input_vol - input_vol.mean(
                        dim=(1, 2, 3), keepdim=True
                    )
                    input_vol = input_vol / (
                        input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                    )
                    input_vol = input_vol.half()
                    with torch.nn.attention.sdpa_kernel(
                        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
                    ):
                        slice_pred = model(input_vol[None])[0]

                    
                    cell_prob[
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += (
                        slice_pred[0].sigmoid() * weight_template
                    )
                    
                    cell_flow[
                        :,
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += (
                        slice_pred[1:] * weight_template
                    )
                    weight[
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += weight_template

            weight += 1e-12
            cell_prob /= weight
            cell_flow /= weight
            cell_prob = (
                cell_prob.cpu()[
                    rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
                ]
                .contiguous()
                .numpy()
            )
            cell_flow = (
                cell_flow.cpu()[
                    :, rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
                ]
                .contiguous()
                .numpy()
            )

            # Assuming ref_arr and sig_arr are torch tensors on GPU from crop_with_pad
            # Convert them to CPU numpy arrays
            ref_np = ref_arr.cpu().numpy()
            sig_np = sig_arr.cpu().numpy()

            z_crop, y_crop, x_crop = ref_np.shape
            trimmed_ref = ref_np[
                z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
                y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
                x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
            ]
            trimmed_sig = sig_np[
                z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
                y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
                x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
            ]

            # Stack them along a new axis so that the shape becomes (2, 40, 256, 256)
            input_image = np.stack([trimmed_ref, trimmed_sig], axis=0)

            cellmask = compute_masks(
                cell_flow[:,4,...],
                cell_prob[4,...],
                min_size=0,
                flow_threshold=None,
                cellprob_threshold=0.5,
                do_3D=False,
            )[0]


            # TODO crop image and results here:
            image_st = []
            if seg_y_input % 256 == 0:
                image_st_y = 0
            else:
                image_st_y = (seg_y - seg_y_input)//2

            if seg_x_input % 256 == 0:
                image_st_x = 0
            
            else:
                image_st_x = (seg_x - seg_x_input)//2
            
            image_st.append(image_st_y)
            image_st.append(image_st_x)
            input_image = input_image[:,:, image_st[0]:image_st[0]+seg_y_input, image_st[1]:image_st[1]+seg_x_input]

            cellmask_pad = np.zeros((vis_z, vis_y, vis_x), dtype=np.uint16)

            cellmask = cellmask[image_st[0]:image_st[0]+seg_y_input, image_st[1]:image_st[1]+seg_x_input]
            cm_y, cm_x = cellmask.shape

            cmp_z_st = vis_z // 2 - 1
            cmp_y_st = (vis_y - cm_y) // 2
            cmp_y_end = cmp_y_st + cm_y
            cmp_x_st = (vis_x - cm_x) // 2
            cmp_x_end = cmp_x_st + cm_x
            cellmask_pad[cmp_z_st, cmp_y_st:cmp_y_end, cmp_x_st:cmp_x_end] = cellmask

            # area rectangle for visualization
            rectangle = [
                [cmp_z_st, cmp_y_st, cmp_x_st],      # top-left
                [cmp_z_st, cmp_y_st, cmp_x_end],     # top-right
                [cmp_z_st, cmp_y_end, cmp_x_end],    # bottom-right
                [cmp_z_st, cmp_y_end, cmp_x_st],     # bottom-left
            ]


            # cellmask_pad[4] = cellmask

            # visualization area:
            # TODO about boundary edge case!


            vis_z_st = z0 + 4 - vis_z//2
            vis_z_end = vis_z_st + vis_z
            vis_y_st = y0 + seg_y // 2 - vis_y//2
            vis_y_end = vis_y_st + vis_y
            vis_x_st = x0 + seg_x // 2 - vis_x//2
            vis_x_end = vis_x_st + vis_x

            #print('zarr path', zarr_path, 'sig_chn', sig_chn, 'z0', z0, 'vis_z_end', vis_z_end, 'y0', y0, 'vis_y_end', vis_y_end, 'x0', x0, 'vis_x_end', vis_x_end)
            fov_block = zarr.open(zarr_path, mode="r")["0"][
                    sig_chn, vis_z_st:vis_z_end , vis_y_st:vis_y_end, vis_x_st:vis_x_end
                ]
            print('fov_block shape:', fov_block.shape)


            print('Completed Segmentation!')
            print(f"Cell mask shape: {cellmask.shape}")
            print(f"Input image shape: {input_image.shape}")
            self.log_output.append("Completed Segmentation!")
            
            # self.log_output.append(f"Cell mask shape: {cellmask.shape}")
            # self.log_output.append(f"Input image shape: {input_image.shape}")



            # --- Save outputs ---
            z_center = z0 + seg_z//2
            y_center = y0 + seg_y//2
            x_center = x0 + seg_x//2

            out_filename_mask = os.path.join(
                output_dir,
                f"mae_2d_sig{sig_chn}_ref{ref_chn}_z{z_center:04d}_y{y_center:04d}_x{x_center:04d}_mask.npy",
            )
            np.save(out_filename_mask, cellmask)

            self.log_output.append("<span style='color:green'>Inference complete! Results saved to:</span>")
            self.log_output.append(out_filename_mask)
            self.log_output.append("===================================")

            # --- show results in a new viewer ---
            self._viewer_results = napari.Viewer()
            self._viewer_results.window._qt_window.setWindowTitle("2D Segmentation Results")

            z_center = input_image.shape[1] // 2

            
            self._viewer_results.add_image(
                fov_block, name="FOV", blending="additive"
            )

            segment_plane = np.zeros((vis_y, vis_x))

            segment_plane[(vis_y - seg_y_input)//2:(vis_y - seg_y_input)//2 + seg_y_input, 
                          (vis_x - seg_x_input)//2:(vis_x - seg_x_input)//2 + seg_x_input] = input_image[1,z_center]


            self._viewer_results.add_image(
                segment_plane, name="segment_plane"
            )
                
            #self._viewer_results.add_image(input_image, channel_axis=0)
            #self._viewer_results.add_image(input_image[1], name="signal channel", colormap="green", blending="additive")

            cell_count = np.max(cellmask_pad)

            self._viewer_results.add_labels(cellmask_pad, name=f"{idx}_MAE_pseudo_{cell_count}")

            # add a rectangle for the segmen area
            self._viewer_results.add_shapes(
            data=[rectangle],
            shape_type='rectangle',
            edge_color='red',
            # face_color='red',
            opacity=0.2, 
            name='fov_box',
            face_color='transparent',  # Make the rectangle transparent
            )
        
        
        # save current meta info to a csv file:
        data = []
        item_copy = self._cropped_points[idx].copy()
        data.append(item_copy)
        df = pd.DataFrame(data)

        if not os.path.exists('./meta_info'):
            os.makedirs('./meta_info')

        df.to_csv('./meta_info/current_points.csv', index=False)

        # # --- update segmented points ---
        # self._cropped_points = {}
        # print(f"current segmented points is: {self._segemented_points}")
    
    def run_inference(self):
        """Collect UI parameters and execute the backend inference routine."""
        # --- Gather inputs from UI ---
        points = self.points_layer.data

        # get new points in the layer
        cur_points = [tuple(p) for p in points]

        # first time, initialize _last_points
        # if not hasattr(self, '_segmented_points'):
        #     self._segemented_points = {}

        old_set = set(self._segemented_points)
        new_set = set(cur_points)
        added = new_set - old_set

        # segment shape handle if not the same size as model input
        seg_z = 10
        seg_y_input = int(self.seg_shape_y_input.text().strip())
        seg_x_input = int(self.seg_shape_x_input.text().strip())

        
        if seg_y_input % 256 == 0:
            seg_y = seg_y_input
            
        else:
            seg_y = seg_y_input//256 * 256 + 256
                

        if seg_x_input % 256 == 0:
            seg_x = seg_x_input

        else:
            seg_x = seg_x_input//256 * 256 + 256


        if not added:
            self.log_output.append("No new points added. Please add new points in the viewer.")
            return
        
        else:
            # update segmented points
            self.log_output.append(f"add new points: {added}")

         

        zarr_path = self.brain_dir_input.text()
        try:
            ref_channel_value = self.ref_channel_combo.currentText()
            sig_channel_value = self.sig_channel_combo.currentText()
            resolution = int(self.resolution_combo.currentText().split()[0])

            ref_chn = int(ref_channel_value)
            sig_chn = int(sig_channel_value)
            
            print("reference channel is:", ref_chn)
            print("signal channel is:", sig_chn)
            print("resolution is:", resolution)

        except Exception:
            print(
                "Error getting channels."
            )
            return
        print('seg_y:', seg_y)
        print('seg_x:', seg_x)
        try:
            #points = self.points_layer.data
            points_list = []
            for point in added:
                z0, y0, x0 = [int((2**resolution)*i) for i in point]
                # make z0 as center of the patch
                z0 -= 4
                y0 -= seg_y // 2
                x0 -= seg_x // 2
                points_list.append([z0, y0, x0])
        
        except Exception:
            print(
                "Error parsing whole brain location. Please input valid integers for z, y, and x."
            )
            return


        output_dir = self.output_dir_input.text()

        # --- Set backend parameters ---
        # For this example, the checkpoint path is hardcoded.


        ckpt = self.model_ckpt.text()
        model_name = "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep10_roi256_ctx1280_layerscale"

        # For simplicity, we assume a single process execution.
        rank = 0
        world_size = 1
        torch.cuda.set_device(rank % torch.cuda.device_count())

        # Load the model from the models_vit module.
        try:
            model = models_vit.__dict__[model_name](out_chans=3)
        except Exception as e:
            print("Error loading model:", e)
            return
        model.half().eval().cuda()
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        )



        for point in points_list:
            z0, y0, x0 = point
            print(f"current segment: z={z0}, y={y0}, x={x0}....")
            # TODO change the zmax, ymax, xmax to the highest resolution shape
            # zmax = z0 + seg_z
            # ymax = y0 + seg_y
            # xmax = x0 + seg_x
            zarr_file = zarr.open(zarr_path, mode='r')
            self.resolution = list(zarr_file.keys())
            za_large = zarr_file['0']
            dask_large = da.from_zarr(za_large)

            # Get the shape of the whole brain image. if it has channel in first dimension
            if len(dask_large.shape) == 4:
                zmax, ymax, xmax = dask_large.shape[1:]
            else:
                zmax, ymax, xmax = dask_large.shape
            self.log_output.append(f"whole brain shape with highest resolution is: {dask_large.shape}")

            print(f"Volume dimensions: zmax={zmax}, ymax={ymax}, xmax={xmax}")

            # Use the segmentation shape provided by the user.
            z, y, x = seg_z, seg_y, seg_x

            # --- Compute model-specific patch sizes and paddings ---
            rz, ry, rx = model.patch_embed_roi.img_size
            cz, cy, cx = model.patch_embed_ctx.img_size
            az, ay, ax = model.patch_embed_ctx.patch_size

            print(f"Model ROI patch size (rz, ry, rx): ({rz}, {ry}, {rx})")
            print(f"Segmentation shape (z, y, x): ({z}, {y}, {x})")

            # Ensure that the segmentation shape is divisible by the ROI patch size.
            try:
                assert (
                    z % rz == 0 and y % ry == 0 and x % rx == 0
                ), "Segmentation shape must be divisible by the ROI patch size."
            except AssertionError as e:
                print(e)
                return

            zpad_head = (cz - rz) // 2 // az * az
            ypad_head = (cy - ry) // 2 // ay * ay
            xpad_head = (cx - rx) // 2 // ax * ax
            zpad_tail = cz - rz - zpad_head
            ypad_tail = cy - ry - ypad_head
            xpad_tail = cx - rx - xpad_head

            def crop_with_pad(chn, z0, y0, x0):
                """Crop the desired volume with padding if necessary."""
                z_st = z0 - zpad_head - rz // 2
                z_ed = z0 + z + zpad_tail + rz // 2
                y_st = y0 - ypad_head - ry // 2
                y_ed = y0 + y + ypad_tail + ry // 2
                x_st = x0 - xpad_head - rx // 2
                x_ed = x0 + x + xpad_tail + rx // 2
                z_slice = slice(max(z_st, 0), min(z_ed, zmax))
                y_slice = slice(max(y_st, 0), min(y_ed, ymax))
                x_slice = slice(max(x_st, 0), min(x_ed, xmax))
                z_pad = (max(-z_st, 0), max(z_ed - zmax, 0))
                y_pad = (max(-y_st, 0), max(y_ed - ymax, 0))
                x_pad = (max(-x_st, 0), max(x_ed - xmax, 0))
                arr = zarr.open(zarr_path, mode="r")["0"][
                    chn, z_slice, y_slice, x_slice
                ]
                if any(a > 0 or b > 0 for a, b in (z_pad, y_pad, x_pad)):
                    arr = np.pad(arr, (z_pad, y_pad, x_pad))
                return arr

            # Crop the region for both channels.
            ref_arr = crop_with_pad(ref_chn, z0, y0, x0)
            sig_arr = crop_with_pad(sig_chn, z0, y0, x0)

            # --- Set up sliding window offsets for segmentation ---
            z_offs_list, y_offs_list, x_offs_list = np.meshgrid(
                np.arange(0, z + 1, rz // 2),
                np.arange(0, y + 1, ry // 2),
                np.arange(0, x + 1, rx // 2),
            )
            z_offs_list = z_offs_list.reshape(-1).tolist()
            y_offs_list = y_offs_list.reshape(-1).tolist()
            x_offs_list = x_offs_list.reshape(-1).tolist()

            # Prepare tensors for accumulating the outputs.
            cell_prob = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
            cell_flow = torch.zeros([2, z + rz, y + ry, x + rx], device="cuda")
            weight = torch.zeros([z + rz, y + ry, x + rx], device="cuda")

            # Create a weight template for blending the patches.
            z_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rz) - 1.0) ** 2
            y_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=ry) - 1.0) ** 2
            x_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rx) - 1.0) ** 2
            weight_template = (
                3.0
                - z_dist_sq.view(-1, 1, 1)
                - y_dist_sq.view(1, -1, 1)
                - x_dist_sq.view(1, 1, -1)
            ) / 3.0
            weight_template = weight_template.cuda()

            # --- Run inference over the sliding window ---
            with torch.no_grad():
                ref_arr = torch.from_numpy(ref_arr).cuda()
                sig_arr = torch.from_numpy(sig_arr).cuda()
                for z_offs, y_offs, x_offs in zip(
                    z_offs_list, y_offs_list, x_offs_list, strict=False
                ):
                    ref_slice = ref_arr[
                        z_offs : z_offs + cz,
                        y_offs : y_offs + cy,
                        x_offs : x_offs + cx,
                    ]
                    sig_slice = sig_arr[
                        z_offs : z_offs + cz,
                        y_offs : y_offs + cy,
                        x_offs : x_offs + cx,
                    ]
                    input_vol = torch.stack([ref_slice, sig_slice], dim=0).float()
                    # Normalize input.
                    input_vol = input_vol - input_vol.mean(
                        dim=(1, 2, 3), keepdim=True
                    )
                    input_vol = input_vol / (
                        input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                    )
                    input_vol = input_vol.half()
                    with torch.nn.attention.sdpa_kernel(
                        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
                    ):
                        slice_pred = model(input_vol[None])[0]

                    
                    cell_prob[
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += (
                        slice_pred[0].sigmoid() * weight_template
                    )
                    
                    cell_flow[
                        :,
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += (
                        slice_pred[1:] * weight_template
                    )
                    weight[
                        z_offs : z_offs + rz,
                        y_offs : y_offs + ry,
                        x_offs : x_offs + rx,
                    ] += weight_template

            weight += 1e-12
            cell_prob /= weight
            cell_flow /= weight
            cell_prob = (
                cell_prob.cpu()[
                    rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
                ]
                .contiguous()
                .numpy()
            )
            cell_flow = (
                cell_flow.cpu()[
                    :, rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
                ]
                .contiguous()
                .numpy()
            )

            # Assuming ref_arr and sig_arr are torch tensors on GPU from crop_with_pad
            # Convert them to CPU numpy arrays
            ref_np = ref_arr.cpu().numpy()
            sig_np = sig_arr.cpu().numpy()

            z_crop, y_crop, x_crop = ref_np.shape
            trimmed_ref = ref_np[
                z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
                y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
                x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
            ]
            trimmed_sig = sig_np[
                z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
                y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
                x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
            ]

            # Stack them along a new axis so that the shape becomes (2, 40, 256, 256)
            input_image = np.stack([trimmed_ref, trimmed_sig], axis=0)

            cellmask = compute_masks(
                cell_flow[:,4,...],
                cell_prob[4,...],
                min_size=0,
                flow_threshold=None,
                cellprob_threshold=0.5,
                do_3D=False,
            )[0]


            # TODO crop image and results here:
            image_st = []
            if seg_y_input % 256 == 0:
                image_st_y = 0
            else:
                image_st_y = (seg_y - seg_y_input)//2

            if seg_x_input % 256 == 0:
                image_st_x = 0
            
            else:
                image_st_x = (seg_x - seg_x_input)//2
            
            image_st.append(image_st_y)
            image_st.append(image_st_x)
            input_image = input_image[:,:, image_st[0]:image_st[0]+seg_y_input, image_st[1]:image_st[1]+seg_x_input]

            # cellmask_pad = np.zeros((10, seg_y_input, seg_x_input), dtype=np.uint16)

            # print('!!!! debug cellmask shape is ', cellmask.shape)
            # cellmask = cellmask[image_st[0]:image_st[0]+seg_y_input, image_st[1]:image_st[1]+seg_x_input]
            #
            vis_z = int(self.vis_shape_z_input.text().strip())
            vis_y = int(self.vis_shape_y_input.text().strip())
            vis_x = int(self.vis_shape_x_input.text().strip())
            cellmask_pad = np.zeros((vis_z, vis_y, vis_x), dtype=np.uint16)

            cellmask = cellmask[image_st[0]:image_st[0]+seg_y_input, image_st[1]:image_st[1]+seg_x_input]
            cm_y, cm_x = cellmask.shape

            cmp_z_st = vis_z // 2 - 1
            cmp_y_st = (vis_y - cm_y) // 2
            cmp_y_end = cmp_y_st + cm_y
            cmp_x_st = (vis_x - cm_x) // 2
            cmp_x_end = cmp_x_st + cm_x
            cellmask_pad[cmp_z_st, cmp_y_st:cmp_y_end, cmp_x_st:cmp_x_end] = cellmask

            # area rectangle for visualization
            rectangle = [
                [cmp_z_st, cmp_y_st, cmp_x_st],      # top-left
                [cmp_z_st, cmp_y_st, cmp_x_end],     # top-right
                [cmp_z_st, cmp_y_end, cmp_x_end],    # bottom-right
                [cmp_z_st, cmp_y_end, cmp_x_st],     # bottom-left
            ]


            # cellmask_pad[4] = cellmask

            # visualization area:
            # TODO about boundary edge case!


            vis_z_st = z0 + 4 - vis_z//2
            vis_z_end = vis_z_st + vis_z
            vis_y_st = y0 + seg_y // 2 - vis_y//2
            vis_y_end = vis_y_st + vis_y
            vis_x_st = x0 + seg_x // 2 - vis_x//2
            vis_x_end = vis_x_st + vis_x

            #print('zarr path', zarr_path, 'sig_chn', sig_chn, 'z0', z0, 'vis_z_end', vis_z_end, 'y0', y0, 'vis_y_end', vis_y_end, 'x0', x0, 'vis_x_end', vis_x_end)
            fov_block = zarr.open(zarr_path, mode="r")["0"][
                    sig_chn, vis_z_st:vis_z_end , vis_y_st:vis_y_end, vis_x_st:vis_x_end
                ]
            print('fov_block shape:', fov_block.shape)


            print('Completed Segmentation!')
            print(f"Cell mask shape: {cellmask.shape}")
            print(f"Input image shape: {input_image.shape}")
            self.log_output.append("Completed Segmentation!")
            
            # self.log_output.append(f"Cell mask shape: {cellmask.shape}")
            # self.log_output.append(f"Input image shape: {input_image.shape}")



            # --- Save outputs ---
            z_center = z0 + seg_z//2
            y_center = y0 + seg_y//2
            x_center = x0 + seg_x//2

            out_filename_mask = os.path.join(
                output_dir,
                f"mae_2d_sig{sig_chn}_ref{ref_chn}_z{z_center:04d}_y{y_center:04d}_x{x_center:04d}_mask.npy",
            )
            np.save(out_filename_mask, cellmask)

            self.log_output.append("<span style='color:green'>Inference complete! Results saved to:</span>")
            self.log_output.append(out_filename_mask)
            self.log_output.append("===================================")

           # --- update segmented points ---
            self._segemented_points +=  [tuple(p) for p in added]
            print(f"current segmented points is: {self._segemented_points}")


            # --- show results in a new viewer ---
            self._viewer_results = napari.Viewer()
            self._viewer_results.window._qt_window.setWindowTitle("2D Segmentation Results")

            z_center = input_image.shape[1] // 2

            self._viewer_results.add_image(
                input_image[1,z_center], name="segment_plane"
            )
            self._viewer_results.add_image(
                fov_block, name="FOV", blending="additive"
            )
                
            #self._viewer_results.add_image(input_image, channel_axis=0)
            #self._viewer_results.add_image(input_image[1], name="signal channel", colormap="green", blending="additive")

            self._viewer_results.add_labels(cellmask_pad, name="cell_mask")

            # add a rectangle for the segmen area
            self._viewer_results.add_shapes(
            data=[rectangle],
            shape_type='rectangle',
            edge_color='red',
            face_color='red',
            opacity=0.2, 
            name='fov_box'
            )



    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
