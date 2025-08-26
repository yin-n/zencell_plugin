# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Example code for inference with MedIM
'''
import sys
sys.path.insert(0, './src/SAM-Med3D')

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
from os.path import join
from glob import glob
from collections import defaultdict
import torch.nn.functional as F
from segment_anything.build_sam3D_cellpose import sam_model_registry3D
#SAM-Med3D.segment_anything.build_sam3D_cellpose import sam_model_registry3D
from cellpose.dynamics import compute_masks
from scipy.ndimage import binary_fill_holes

IMAGE_SIZE = 32

def build_cp_model(checkpoint=None):
    """
    Build a SAM-Med3D model with Cellpose decoder
    """
    return sam_model_registry3D["vit_b_ori_cellpose"](checkpoint=checkpoint)


def random_sample_next_click(prev_mask, gt_mask):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if (not true_masks.any()):
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True
    else:
        is_positive = False

    sampled_point = point.clone().detach().reshape(1, 1, 3)
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    prompt_generator=random_sample_next_click,
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

    # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using device", device)
    model = model.to(device)
    
    # import pdb; pdb.set_trace()

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0,
                                                   3).to(device), torch.zeros(
                                                       1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64).to(device)
        prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
        prev_low_res_mask = F.interpolate(prev_low_res_mask,
                                              size=(roi_image.shape[2] // 4, roi_image.shape[3] // 4, roi_image.shape[4] // 4),
                                              mode='nearest').to(torch.float32)
        if (roi_gt is not None):
            print("!!! [Debug @sam_model_infer] roi_gt is", roi_gt) 
           
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        print("!!! [Debug @sam_model_infer] points_coords is", points_coords)  # (1, 1, 3)
        print("!!! [Debug @sam_model_infer] points_labels shape is", points_labels)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask.to(device),
            # masks=None,
        )

        low_res_masks, flow_z, flow_y, flow_x, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
            multimask_output=False,  # we only need one mask

        )

      
        prev_mask = low_res_masks

    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
    # !!! debug
    # downsample to 32x32x32
    #medsam_seg_prob= F.interpolate(medsam_seg_prob, size=(32, 32, 32), mode='trilinear', align_corners=False)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()

    flow_y = flow_y.cpu().numpy().squeeze()
    flow_x = flow_x.cpu().numpy().squeeze()

    dP = np.zeros((2, 32, 32), dtype=np.float32)
    #dP[0] = flow_z.astype(np.float32)
    dP[0]= flow_y[16].astype(np.float32)
    dP[1] = flow_x[16].astype(np.float32)
    cellprob = medsam_seg_prob[16].astype(np.float32)

    medsam_seg_mask = np.zeros((32, 32, 32), dtype=np.float32)

    seg_mask = compute_masks(dP, cellprob, min_size=0, flow_threshold=None, cellprob_threshold = 0.5, do_3D=False)

    # [CAHNGE]
    # here add a post-processing step to fill holes in the mask
    # Get all labels (excluding background 0)
    labels = np.unique(seg_mask)
    labels = labels[labels != 0]

    # Create empty result mask
    filled_mask = np.zeros_like(seg_mask)

    for label in labels:
        region = (seg_mask == label)
        filled_region = binary_fill_holes(region)
        filled_mask[filled_region] = label

    medsam_seg_mask[16] = filled_mask
    return medsam_seg_mask


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def resample_nii(imgs: np.array,
                 gts: np.array,
                 prev_seg: np.array,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                ):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(
                    image=tio.ScalarImage(tensor=imgs[None]), 
                    label=tio.LabelMap(tensor=gts[None]),
                    prev_seg=tio.LabelMap(tensor=prev_seg[None]),
                     )
    # change no need to resample the image
    # resampler = tio.Resample(target=target_spacing)
    # resampled_subject = resampler(subject)
    return subject


def read_data_from_subject(subject):

    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)

    # import pdb; pdb.set_trace()
    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    
    
    prev_seg3D_roi = subject_roi.prev_seg.data.clone().detach().unsqueeze(1)
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + IMAGE_SIZE - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + IMAGE_SIZE - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + IMAGE_SIZE - padding_params[4] - padding_params[5],
    )

    # debug!!!
    # upscale the image to 128x128x128
    print("!!![Debug] @read_data_from_subject img3D_roi shape is", img3D_roi.shape)  # [1, 1, 32, 32, 32]
    img3D_roi_upscale = F.interpolate(img3D_roi, size=(128, 128, 128), mode='trilinear', align_corners=False)
    print("!!![Debug] @read_data_from_subject img3D_roi_upscale shape is", img3D_roi_upscale.shape)  # [1, 1, 32, 32, 32]

    gt3D_roi_upscale = F.interpolate(gt3D_roi, size=(128, 128, 128), mode='nearest')  # for labels



    meta_info = {
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi_upscale, # [Change] upscale to 128x128x128
        gt3D_roi_upscale,
        prev_seg3D_roi,
        meta_info,
    )


def data_preprocess(imgs, cls_gt, cls_prev_seg, orig_spacing, category_index):
    subject = resample_nii(imgs, cls_gt, cls_prev_seg, target_spacing=[t/o for o, t in zip(orig_spacing, [1, 1, 1])]) # [Change] change this spacing to [1, 1, 1]
    roi_image, roi_label, roi_prev_seg, meta_info = read_data_from_subject(subject)
    
    meta_info["orig_shape"] = imgs.shape
    meta_info["resampled_shape"] = subject.spatial_shape,
    return roi_image, roi_label, roi_prev_seg, meta_info


def data_postprocess(roi_pred, meta_info, output_dir='outputs'):

    # !!! debugg
    print("roi_pred shape is", roi_pred.shape) # 32,32,32
    #np.save('roi_pred.npy', roi_pred)
    os.makedirs(output_dir, exist_ok=True)
    pred3D_full = np.zeros(*meta_info["resampled_shape"])
    padding_params = meta_info["padding_params"]
    unpadded_pred = roi_pred[padding_params[0] : IMAGE_SIZE-padding_params[1],
                             padding_params[2] : IMAGE_SIZE-padding_params[3],
                             padding_params[4] : IMAGE_SIZE-padding_params[5]]
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = unpadded_pred
    #print('meta_info is ', meta_info)
   
    pred3D_full_ori = pred3D_full
   
    return pred3D_full_ori


def read_data(img, clicks):

    # spacing = sitk_spacing
    spacing = [1, 1, 1] #[Change] change this spacing from [1.5, 1.5, 1.5] to [1, 1, 1]
    # z-score normalize imgs
    img = img.astype(np.float32)
    # parsing boxes/clicks tensor, allow category to has more than 1 clicks
    all_clicks = defaultdict(list)
    prev_pred = np.zeros_like(img, dtype=np.uint8)
    
    if (clicks is not None):
        for cls_idx, cls_click_dict in enumerate(clicks):
            for click in cls_click_dict['fg']:
            #     all_clicks[cls_idx].append(((click[2], click[1], click[0]), [1]))
                all_clicks[cls_idx].append((click, [1]))
            for click in cls_click_dict['bg']:
                all_clicks[cls_idx].append((click, [0]))

    return img, spacing, all_clicks, prev_pred


def read_data_from_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    imgs = data.get('imgs', None)

    sitk_spacing = data.get('spacing', None)
    # spacing = sitk_spacing
    spacing = [1, 1, 1]  # [Change] change this spacing from [1.5, 1.5, 1.5] to [1, 1, 1]

    # z-score normalize imgs
    imgs = imgs.astype(np.float32)

    # parsing boxes/clicks tensor, allow category to has more than 1 clicks
    all_clicks = defaultdict(list)
    # get bbox click first
    boxes = data.get('boxes', None)
    if (boxes is not None):
        for cls_idx, bbox in enumerate(boxes):
            all_clicks[cls_idx].append((
            ((bbox['z_min']+bbox['z_max'])/2, (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,), # center of bbox
            [1], # positive click
            ))
       
    # get point click then
    prev_pred = data.get('prev_pred', np.zeros_like(imgs, dtype=np.uint8))
    clicks = data.get('clicks', None)
    if (clicks is not None):
        for cls_idx, cls_click_dict in enumerate(clicks):
            for click in cls_click_dict['fg']:
            #     all_clicks[cls_idx].append(((click[2], click[1], click[0]), [1]))
                all_clicks[cls_idx].append((click, [1]))
            for click in cls_click_dict['bg']:
                all_clicks[cls_idx].append((click, [0]))


    # import pdb; pdb.set_trace()
    return imgs, spacing, all_clicks, prev_pred


def create_gt_arr(shape, point, category_index, square_size=2):
    # Create an empty array with the same shape as the input array
    gt_array = np.zeros(shape)
    
    # Extract the coordinates of the point
    z, y, x = point
    print('@create_gt_arr: point is', point, 'category_index is', category_index)
    
    # Calculate the half size of the square
    half_size = square_size // 2
    
    # # Calculate the coordinates of the square around the point
    # z_min = max(int(z - half_size), 0)
    # z_max = min(int(z + half_size) + 1, shape[0])
    # y_min = max(int(y - half_size), 0)
    # y_max = min(int(y + half_size) + 1, shape[1])
    # x_min = max(int(x - half_size), 0)
    # x_max = min(int(x + half_size) + 1, shape[2])
    
    # Set the values within the square to 1
    #gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index
    z_round =  np.round(z).astype(int)
    y_round =  np.round(y).astype(int)
    x_round =  np.round(x).astype(int)

    gt_array[z_round, y_round, x_round] = category_index  # Ensure the point itself is set
    #gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index  # Set the square area to category_index
    return gt_array


if __name__ == "__main__":
        # load model
        ckpt_path =  "/mnt/aperto/yin/sammed3d_ckpt/img_32/sam_model_latest.pth"
        # model = medim.create_model("SAM-Med3D",
        #                             pretrained=True,
        #                             checkpoint_path=ckpt_path)

        #sam_model = sam_model_registry3D["vit_b_ori_cellpose"](checkpoint=ckpt_path)

        sam_model = build_cp_model(checkpoint=ckpt_path)
        # output dir
        out_dir = "/mnt/aperto/yin/sammed_3D/sample_20/res"

        # img and clicks
        img = np.load('/mnt/aperto/yin/sammed_3D/sample_20/img/test_img.npz')['imgs']

        clicks = [
            {
                'fg': np.array([[12, 390, 142]]), 
                'bg': np.array([], dtype=np.int32).reshape(0, 3)  
            }
        ]

        
        img, spacing, all_clicks, prev_pred = read_data(img, clicks)
        final_pred = np.zeros_like(img, dtype=np.uint8)

        # prediction process
        for idx, cls_clicks in all_clicks.items():
                category_index = idx + 1
                pred_ori = prev_pred==category_index
                final_pred[pred_ori!=0] = category_index
                if (cls_clicks[-1][1][0] == 1):
                    cls_gt = create_gt_arr(img.shape, cls_clicks[-1][0], category_index=category_index)
                    # print(category_index, imgs.shape, spacing, cls_clicks, (cls_gt==category_index).sum())
                    # continue
                    cls_prev_seg = prev_pred==category_index
                    roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(img, cls_gt, cls_prev_seg,
                                                                    orig_spacing=spacing, 
                                                                    category_index=category_index)

                    ''' 3. infer with the pre-trained SAM-Med3D model '''
                    roi_pred = sam_model_infer(sam_model, roi_image, roi_gt=None, prev_low_res_mask=roi_prev_seg) #[Change] change roi_gt from roi_label to None
     

                    ''' 4. post-process and save the result '''
                    pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
                    final_pred[pred_ori!=0] = category_index

        output_path = osp.join(out_dir,'test_cp.npz')
        np.savez_compressed(output_path, segs=final_pred)
        print("result saved to", output_path)