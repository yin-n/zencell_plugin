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

        self.layout = QtWidgets.QVBoxLayout()

        self.data_shape = [12, 6000, 8000]
        self.resolution = ['0','1','2']

        # self.setLayout(QHBoxLayout())

        # 1. Whole brain (zarr) path.
        self.whole_brain_label = QtWidgets.QLabel("Whole Brain Path (zarr):")
        self.whole_brain_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.whole_brain_label)
        self.layout.addWidget(self.whole_brain_input)

        #  display metadata and automatically show the channels button
        self.meta_button = QtWidgets.QPushButton("get metadata")
        self.meta_button.clicked.connect(self.get_metadata)
        self.layout.addWidget(self.meta_button)



        # 3. Reference and signal channels.
        self.ref_channel_label = QtWidgets.QLabel(
            "Reference Channel:"
        )
    
        self.layout.addWidget(self.ref_channel_label)
        
        self.ref_channel_combo = QtWidgets.QComboBox()
        self.ref_channel_combo.addItems([str(i + 1) for i in range(self.data_shape[0])])
        self.layout.addWidget(self.ref_channel_combo)


        self.sig_channel_label = QtWidgets.QLabel(
            "Signal Channel:"
        )
    
        self.layout.addWidget(self.sig_channel_label)
        self.sig_channel_combo = QtWidgets.QComboBox()
        self.sig_channel_combo.addItems([str(i + 1) for i in range(self.data_shape[0])])
        self.layout.addWidget(self.sig_channel_combo)


        # 4. Resolution selection.
        self.resolution_label = QtWidgets.QLabel(
            "Resolution:"
        )
    
        self.layout.addWidget(self.resolution_label)
        
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems(self.resolution)
        self.layout.addWidget(self.resolution_combo)

        # 5. A button to show whole brain.
        self.show_whole_brain_button = QtWidgets.QPushButton("Show Whole Brain")
        self.show_whole_brain_button.clicked.connect(self.show_whole_brain)
        self.layout.addWidget(self.show_whole_brain_button)


        # 6. Output directory with a browse button.
        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QtWidgets.QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)
        self.layout.addWidget(self.output_dir_label)
        self.layout.addLayout(output_dir_layout)

        # 7. Model selection.
        self.model_label = QtWidgets.QLabel("Model Path (.pth):")
        self.model_ckpt = QtWidgets.QLineEdit()
        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model_ckpt)

        # Run Inference button.

        self.run_button = QtWidgets.QPushButton("Run Inference")
        self.run_button.clicked.connect(self.run_inference)
        # self.run_button.clicked.connect(self._on_click)
        self.layout.addWidget(self.run_button)



        # metadata button
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)


        self.setLayout(self.layout)
        self.setup_defaults()
    
    # TODO set default path
    def setup_defaults(self):
        self.whole_brain_input.setText("/mnt/aperto/tatz_brain_data/240620_01_MX007-1/fused.zarr")  
        self.model_ckpt.setText("/mnt/aperto/yin/zencell_ckpt/2D/checkpoint-499.pth")

    def get_metadata(self): 
        zarr_path = self.whole_brain_input.text()
        self.log_output.append(f"reading zarr file path: {zarr_path}")
        
        try:
            zarr_file = zarr.open(zarr_path, mode='r')
            self.resolution = list(zarr_file.keys())
            za_wh = zarr_file[self.resolution[-1]]
            dask_wh = da.from_zarr(za_wh)

            self.data_shape = dask_wh.shape
            self.log_output.append(f"whole brain shape is: {self.data_shape}")

            # read metadata info
            attrs = zarr_file.attrs.asdict()
            if "multiscales" in attrs:
                import json
                multiscales = attrs["multiscales"]
                self.log_output.append(f"multiscales: {json.dumps(multiscales, indent=2)}")
            else:
                self.log_output.append("No multiscales metadata found.")

        except Exception as e:
            self.log_output.append(f"<span style='color:red'>reading metadata exception: {e}</span>")
            return


        # update the resolution combobox
        self.resolution_combo.clear()
        self.resolution_combo.addItems(self.resolution)
        self.log_output.append("update resolution combobox by automatically read metadata")
        #  update the shape of the data
        self.log_output.append(f"update shape of the data by automatically read metadata: {self.data_shape}")



        #  update comboboxes for reference and signal channels
        num_channels = self.data_shape[0]

        self.ref_channel_combo.clear()
        self.ref_channel_combo.addItems([str(i + 1) for i in range(num_channels)])

        self.sig_channel_combo.clear()
        self.sig_channel_combo.addItems([str(i + 1) for i in range(num_channels)])

        self.log_output.append("update selected ref/sig channels by automatically read metadata")


        
        self.log_output.moveCursor(QtGui.QTextCursor.End) 


    def show_whole_brain(self):
        # --- Gather inputs from UI ---
        zarr_path = self.whole_brain_input.text()
        try:
            ref_channel_value = self.ref_channel_combo.currentText()
            sig_channel_value = self.sig_channel_combo.currentText()
            resolution = self.resolution_combo.currentText()

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
            edge_color='pink',
            face_color='pink',
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



    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

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

        if not added:
            print("No new points added. Please add new points in the viewer.")
            return
        
        else:
            # update segmented points
            self.log_output.append(f"add new points: {added}")

         

        zarr_path = self.whole_brain_input.text()
        try:
            ref_channel_value = self.ref_channel_combo.currentText()
            sig_channel_value = self.sig_channel_combo.currentText()
            resolution = int(self.resolution_combo.currentText())

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

        try:
            #points = self.points_layer.data
            points_list = []
            for point in added:
                z0, y0, x0 = [int((2**resolution)*i) for i in point]
                # make z0 as center of the patch
                z0 -= 4
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

        seg_z = 10
        seg_y = 256
        seg_x = 256

        for point in points_list:
            z0, y0, x0 = point
            self.log_output.append(f"current segment: z={z0}, y={y0}, x={x0}....")
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

            # --- Save outputs ---
            out_filename_prob = os.path.join(
                output_dir,
                f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_prob.npy",
            )
            out_filename_flow = os.path.join(
                output_dir,
                f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_flow.npy",
            )
            np.save(out_filename_prob, cell_prob)
            np.save(out_filename_flow, cell_flow)

            print("Inference complete. Results saved to:")
            print(out_filename_prob)
            print(out_filename_flow)

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

            # Stack them along a new axis so that the shape becomes (2, 40, 1024, 1024)
            input_image = np.stack([trimmed_ref, trimmed_sig], axis=0)

            cellmask = compute_masks(
                cell_flow[:,4,...],
                cell_prob[4,...],
                min_size=0,
                flow_threshold=None,
                cellprob_threshold=0.5,
                do_3D=False,
            )[0]
            print('Completed Segmentation!')
            print(f"Cell mask shape: {cellmask.shape}")
            print(f"Input image shape: {input_image.shape}")
            self.log_output.append("Completed Segmentation!")
            self.log_output.append(f"Cell mask shape: {cellmask.shape}")
            self.log_output.append(f"Input image shape: {input_image.shape}")

            # update segmented points
            self._segemented_points +=  [tuple(p) for p in added]
            self.log_output.append(f"current segmented points is: {self._segemented_points}")

            # Now you can display it in napari as a multichannel image:
            
            self._viewer_results = napari.Viewer()
            self._viewer_results.window._qt_window.setWindowTitle("2D Segmentation Results")
                
            self._viewer_results.add_image(input_image[0,4,...], name="reference channel", colormap="blue", blending="additive")
            self._viewer_results.add_image(input_image[1,4,...], name="signal channel", colormap="green", blending="additive")

            self._viewer_results.add_labels(cellmask, name="Cell Mask")
            self._viewer_results.show(block=True)


    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
