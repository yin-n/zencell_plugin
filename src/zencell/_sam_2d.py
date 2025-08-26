import medim
import numpy as np

import sys
import os
# add med sam path
# sam_med3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SAM-Med3D"))
# # sys.path.append(sam_med3d_path)

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAM-Med3D')))

import sys
sys.path.append('/mnt/aperto/yin/zencell_plugin/src/SAM-Med3D')

import dask.array as da
import napari
import numpy as np
import torch
import zarr
from app_model.backends.qt import QMenuItemAction
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QComboBox, QVBoxLayout
from PyQt5 import QtGui  
import re
import pandas as pd

from medim_infer_cp_2d import create_gt_arr
from medim_infer_cp_2d import sam_model_infer
from medim_infer_cp_2d import data_postprocess
from medim_infer_cp_2d import data_preprocess
from medim_infer_cp_2d import read_data
from medim_infer_cp_2d import build_cp_model
import tifffile as tiff

import os.path as osp
from collections import defaultdict

class SAMQWidget2D(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.points_layer = None
        self._last_points = []
        self._segemented_points = []
        self._largest_instance = 0

        self.layout = QtWidgets.QVBoxLayout()

        # 1. log 
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        
        # 2. select image part
        self.image_layer = None 
        self.dropdown = QComboBox()

        if self.has_image_layer(self._viewer):
            # check if there is an image layer
            self.log_output.append("initial image layer found")
            for layer in self._viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    self.dropdown.addItem(layer.name)
                    print(f"Image layer found: {layer.name}")
    
        
        # viewer listen to layer added and removed events
        self._viewer.layers.events.inserted.connect(self.on_layer_added)
        self._viewer.layers.events.removed.connect(self.on_layer_removed)
        self.dropdown.currentIndexChanged.connect(self.on_selection_change)
        self.layout.addWidget(self.dropdown)

        # 3. create a points layer
        self.points_layer = self.points_layer = self._viewer.add_points(
            np.empty((0, 3)),  # set as 3D points
            name='click area center for segmentation',
            size=3
        )
        # make points as int
        self.points_layer.data = self.points_layer.data.astype(int)
        self.points_layer.events.data.connect(self._on_points_changed)
        self.log_output.append("add points to click area center for prompt segmentation")
       
        # add a buttton to choose model for compare the performance
        #--------------------------------------------------------------------
        # add parts to load pseudo segmentation
        #--------------------------------------------------------------------
        # self.model_select_label = QtWidgets.QLabel("Model checkpoint file:")
        # self.model_select_input = QtWidgets.QLineEdit()
        # self.browse_model_button = QtWidgets.QPushButton("Browse")
        # self.browse_model_button.clicked.connect(self.browse_model_path)

        # model_load_layout = QtWidgets.QHBoxLayout()
        # model_load_layout.addWidget(self.model_select_input)
        # model_load_layout.addWidget(self.browse_model_button)

        # self.layout.addWidget(self.model_select_label)
        # self.layout.addLayout(model_load_layout)

        #--------------------------------------------------------------------

        # 4. Saving area
        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QtWidgets.QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)
        self.layout.addWidget(self.output_dir_label)
        self.layout.addLayout(output_dir_layout)


        self.save_button = QtWidgets.QPushButton("combine and save!")
        self.save_button.clicked.connect(self.combine_and_save)


        # 5. Run inference area
        self.run_button = QtWidgets.QPushButton("predict!")
        self.run_button.clicked.connect(self.run_inference)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.save_button)

        self.layout.addWidget(self.log_output) # add log at the end
        self.setLayout(self.layout)
        


    def has_image_layer(self, viewer) -> bool:
        return any(isinstance(layer, napari.layers.Image) for layer in viewer.layers)
    


    def on_layer_added(self, event):
    
        layer = event.value
        if layer.__class__.__name__ == 'Image':
            print(f"New image layer added: {layer.name}")
            self.dropdown.addItem(layer.name)

    def on_layer_removed(self, event):
        layer = event.value
        if layer.__class__.__name__ == 'Image':
            print(f"Image layer removed: {layer.name}")
            # find the deleted layer in the dropdown and remove it
            index = self.dropdown.findText(layer.name)
            if index != -1:
                self.dropdown.removeItem(index)

    def on_selection_change(self, index):
        selected_name = self.dropdown.currentText()
        if selected_name in self._viewer.layers:
            self.image_layer = self._viewer.layers[selected_name]
            print(f"Selected image layer: {selected_name}")
        else:
            self.image_layer = None


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


    def run_inference(self):
        """Collect UI parameters and execute the backend inference routine."""
        # --- Gather inputs from UI ---
        points = self.points_layer.data

        # get new points in the layer
        cur_points = [tuple(p) for p in points]

        old_set = set(self._segemented_points)
        new_set = set(cur_points)
        added = new_set - old_set

        if not added:
            print("No new points added. Please add new points in the viewer.")
            return
        
        else:
            # update segmented points
            self.log_output.append(f"add new points: {added}")
        

        # --- Run the backend inference routine ---
        img = self.image_layer.data

        points_list = []
        clicks = []
        for point in added:
            z0, y0, x0 = point
            points_list.append([z0, y0, x0])

        for item in points_list:
            print('item', item)
            points_dict = {
                'fg': np.array([item]),  # foreground point
                'bg': np.array([], dtype=np.int32).reshape(0, 3)  # background points
            }
            clicks.append(points_dict)

    ## CLICK MUST BE LIKE THIS FORMAT
    #     clicks = [
    #     {
    #         'fg': np.array([[49.        , 243.25277685, 232.56888284]]),
    #         'bg': [(5, 10, 15)]
    #     },
    #     {
    #         'fg': np.array([[49.        , 281.21176995, 147.71936886]]),
    #         'bg': [(35, 45, 55), (30, 40, 50)]
    #     }
    # ]


        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/sam_model_dice_best.pth"
        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/flow10_ep300/sam_model_dice_best.pth"
        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32/sam_model_dice_best.pth"
        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32/sam_model_dice_best.pth"
        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32_d3/sam_model_dice_best.pth"
        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/2d_cp_lr_2e_4/sam_model_40_ckp.pth"

        ckpt_path = '/mnt/aperto/yin/sammed3d_ckpt/2d_cp_v2_lr_2e_4/sam_model_40_ckp.pth'

        # ckpt_path = self.model_select_input.text()
        # if not ckpt_path:
        #     QtWidgets.QMessageBox.warning(self, "Warning", "Please select a model checkpoint file.")
        #     return
        # print('ckpt_path', ckpt_path)
        sam_model = build_cp_model(checkpoint=ckpt_path)
    
        out_dir = self.output_dir_input.text()
        img, spacing, all_clicks, prev_pred = read_data(img, clicks)
        spacing = [1, 1, 1] 
        #print('all_clicks', all_clicks) 
        final_pred = np.zeros_like(img, dtype=np.uint8)
        for idx, cls_clicks in all_clicks.items():
                category_index = idx + 1 + self._largest_instance
                pred_ori = prev_pred==category_index
                final_pred[pred_ori!=0] = category_index
                # print('!!!idx', idx)
                # print('!!!cls_clicks', cls_clicks)
               
                if (cls_clicks[0][1][0] == 1):
                    cls_gt = create_gt_arr(img.shape, cls_clicks[0][0], category_index=category_index, square_size=2)
                    
                    # continue
                    cls_prev_seg = prev_pred==category_index
                    roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(img, cls_gt, cls_prev_seg,
                                                                    orig_spacing=spacing, 
                                                                    category_index=category_index)

                    ''' 3. infer with the pre-trained SAM-Med3D model '''
                    roi_pred = sam_model_infer(sam_model, roi_image, roi_gt=None, prev_low_res_mask=roi_prev_seg)#[Change] change roi_gt from roi_label to None

                    ''' 4. post-process and save the result '''
                    pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
                    final_pred[pred_ori!=0] = category_index
        
        self._largest_instance = np.max(final_pred)

        output_path = osp.join(out_dir,'test.npy')
        np.save(output_path, final_pred)
        print("result saved to", output_path)
        # show results
        self._viewer.add_labels(final_pred, name='sam_res', blending ='additive')
        # update segmented points
        self._segemented_points +=  [tuple(p) for p in added]
        self.log_output.append(f"current segmented points is: {self._segemented_points}")
    
    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def browse_model_path(self):

        """Open a file selection dialog and update the model file field."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pth);;All Files (*)"
        )
        if file_path:
            self.model_select_input.setText(file_path)
        
       
    
    def combine_and_save(self):
        """Combine all the segmented points and save them to the specified output directory."""
        output_dir = self.output_dir_input.text()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return


        # TODO combined with MAE results

        mae_pattern = re.compile(r'^\d+_MAE_pseudo_\d+_\d+_\d+$')
        mae_pseudo_layer = None
        for layer in self._viewer.layers:
            if hasattr(layer, 'name') and mae_pattern.match(layer.name):
                mae_pseudo_layer = layer
                break
        
        if mae_pseudo_layer is not None:

            mae_data = mae_pseudo_layer.data.copy()
            mae_data[mae_data > 0] += self._largest_instance
       

        # combine all SAM layers
        count = 0
        combined_sam_layers = None
        sam_pattern = re.compile(r'sam_res')

        for layer in self._viewer.layers:
            if hasattr(layer, 'name') and sam_pattern.match(layer.name):

                count += 1
                if count == 1:
                    combined_sam_layers = layer.data.copy()
                else:
                    sam_data = layer.data.copy()
                    combined_sam_layers += sam_data
        # if some area is overlapped, we will keep the largest instance
        if combined_sam_layers is not None:
            if mae_pseudo_layer is not None:
                combined_sam_layers = np.maximum(combined_sam_layers, mae_data)

      
        # TODO save information to a file
        mae_pseudo_layer_shape = mae_pseudo_layer.data.shape
        z_center = int(mae_pseudo_layer.name.split('_')[-3])
        y_patch = int(mae_pseudo_layer.name.split('_')[-2])
        x_patch = int(mae_pseudo_layer.name.split('_')[-1])
                    
        y_center = mae_pseudo_layer_shape[1] // 2
        x_center = mae_pseudo_layer_shape[2] // 2

        idx = int(mae_pseudo_layer.name.split('_')[0])
        file_path = os.path.join(output_dir, f"{idx}_combined_res.tif")
        tiff.imwrite(file_path, combined_sam_layers[z_center,y_center - y_patch: y_center + y_patch, x_center - x_patch: x_center + x_patch])
        #np.save(output_file, combined_sam_layers)
        self.log_output.append(f"Combined points saved to: {file_path}")

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

        tiff.imwrite(image_tif_path, img)
    
        # TODO save this use coordinates!
        cur_df = pd.read_csv("./meta_info/current_points.csv")

        folder = output_dir
        csv_path = os.path.join(folder, "all_points.csv")
        all_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=cur_df.columns)
        all_csv = pd.concat([all_csv, cur_df], ignore_index=True, sort=False)
        all_csv.to_csv(csv_path, index=False)

        total_counts = len(np.unique(combined_sam_layers)) - 1  # subtract 1 for background
        self.log_output.append(f"Total unique instances in combined result: {total_counts}")

        self._viewer.add_labels(combined_sam_layers, name=f'combined_res_{total_counts}', blending='additive')

if __name__ == "__main__":
    print('test')