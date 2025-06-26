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


from medim_infer_cp import create_gt_arr
from medim_infer_cp import sam_model_infer
from medim_infer_cp import data_postprocess
from medim_infer_cp import data_preprocess
from medim_infer_cp import read_data
from medim_infer_cp import build_cp_model
from cellpose.dynamics import compute_masks

import os.path as osp
from collections import defaultdict

class SAMQWidget3D(QWidget):
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
        self.points_layer.events.data.connect(self._on_points_changed)
        self.log_output.append("add points to click area center for prompt segmentation")
       

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
        ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32_m10_ndec/sam_model_dice_best.pth"

        #ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32_d3_ndec/sam_model_179_step_dice:0.9827064871788025_best.pth"
        # model = medim.create_model("SAM-Med3D",
        #                             pretrained=True,
        #                             checkpoint_path=ckpt_path)

        sam_model = build_cp_model(checkpoint=ckpt_path)
       
        # model = medim.create_model("SAM-Med3D",
        #                               pretrained=True,
        #                               checkpoint_path=ckpt_path)


        out_dir = self.output_dir_input.text()
        img, spacing, all_clicks, prev_pred = read_data(img, clicks)
        spacing = [1, 1, 1] 
        print('spacing is: ', spacing)
        print('all_clicks', all_clicks) 
        final_pred = np.zeros_like(img, dtype=np.uint8)
        for idx, cls_clicks in all_clicks.items():
                category_index = idx + 1 + self._largest_instance
                pred_ori = prev_pred==category_index
                final_pred[pred_ori!=0] = category_index
                print('!!!idx', idx)
                print('!!!cls_clicks', cls_clicks)
               
                if (cls_clicks[0][1][0] == 1):
                    cls_gt = create_gt_arr(img.shape, cls_clicks[0][0], category_index=category_index)
                    
                    # continue
                    cls_prev_seg = prev_pred==category_index
                    roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(img, cls_gt, cls_prev_seg,
                                                                    orig_spacing=spacing, 
                                                                    category_index=category_index)

                    ''' 3. infer with the pre-trained SAM-Med3D model '''
                    roi_pred = sam_model_infer(sam_model, roi_image, roi_gt=roi_label, prev_low_res_mask=roi_prev_seg)#[Change] change roi_gt from roi_label to None

                    ''' 4. post-process and save the result '''
                    pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
                    final_pred[pred_ori!=0] = category_index
        
        self._largest_instance = np.max(final_pred)

        output_path = osp.join(out_dir,'test.npy')
        np.save(output_path, final_pred)
        print("result saved to", output_path)
        # show results
        self._viewer.add_labels(final_pred, name='segmentation result', blending ='additive')

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
    
    def combine_and_save(self):
        """Combine all the segmented points and save them to the specified output directory."""
        output_dir = self.output_dir_input.text()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return
     
        combined_labels = None
       # combine all label layers
        count = 0
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                count += 1
                if count == 1:
                    combined_labels = layer.data.copy()
                else:
                    label_data = layer.data
                    combined_labels += label_data

        # Save the combined points to a file
        # TODO save this use coordinates!
        output_file = os.path.join(output_dir, "segmented_res.npy")
        np.save(output_file, combined_labels)
        self.log_output.append(f"Combined points saved to: {output_file}")
    


if __name__ == "__main__":
    print('test')