""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2025 by the authors.
This code is licensed (see LICENSE for details)

The file contains a U-ARE-ME class wrapper

"""

"""
INIT:       - img -> numpy array (any image of desired size)
            - b_kappa -> bool (optional)
            - kappa_theshold -> float (optional)
            - b_multiframe -> bool (optional)
            - b_robust -> bool (optional)
            - window_length -> int (optional)
            - interframe_sigma -> float (optional)

CALL: rotation, normals, confidence = UAREME.run(img -> numpy array)

If using multiframe input images must be sequential

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torchvision import transforms
from utils.MNMAoptimiser import MNMAoptimiser
from utils import input as input_utils
import uareme.src.utils.visualisation as vis_utils
import cv2
import time


class UAREME():
    def __init__(self,
                 b_trt_model : bool = False,
                 b_kappa : bool = True,
                 kappa_threshold : float = 75.0,
                 b_multiframe : bool = True,
                 b_robust : bool = True,
                 window_length : int = 30,
                 interframe_sigma : float = 0.75):

        self.use_multi = b_multiframe
        if self.use_multi:
            from utils import MFOpt as mfOpt_utils
            self.mf_window = window_length
            self.mf_robustify = b_robust
            self.mf_inter_sigma = interframe_sigma

            self.multiframe_optimiser = mfOpt_utils.gtsam_rot(window_length=self.mf_window, 
                                                              interframe_sigma=self.mf_inter_sigma, 
                                                              robust=self.mf_robustify)

        self.use_kappa = b_kappa
        self.kappa_thresh = kappa_threshold

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = input_utils.define_model(self.device, b_trt_model)
        self.normalise_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.R_opt = np.eye(3)

        self.singleframe_optimiser = MNMAoptimiser(use_kappa=self.use_kappa)
    
        self.MODES_DICT = {1: 'RGB', 2: 'Normals', 3: 'Confidence'}
        self.DISPLAY_MODE = self.MODES_DICT[1]
        self.TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX                                        # List of displays
        self.TITLE_WIDTHS = {self.MODES_DICT[i]: cv2.getTextSize(t, self.TITLE_FONT, 1, 2)[0][0] 
                        + 20 for i, t in self.MODES_DICT.items()}                                            # Display title width 
        self.DISPLAY_WIDTH = 56
        self.prev_frame_time = time.time()


    def run(self, img : np.ndarray, format: str = 'RGB'):
        # Preprocess image to torch format. The input image must be uint8 format
        img_torch = self.preprocess_img(img, format)

        ####################################################################################
        # Run image through network
        if img_torch is not None:
            with torch.no_grad():
                model_out = self.model(img_torch)[0]
                pred_norm = model_out[:, :3, :, :]  # (1, 3, H, W)
                norm_out = pred_norm.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()  # (H, W, 3)
                pred_kappa = model_out[:, 3:, :, :]  # (1, 1, H, W)
                kappa_out = pred_kappa.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()  # (H, W, 1)
                pred_kappa[pred_kappa > self.kappa_thresh] = self.kappa_thresh
                pred_norm_vec = pred_norm[0,...].view(3, -1)  # (3, H*W)
                pred_kappa_vec = pred_kappa[0,...].view(1, -1)  # (1, H*W) 
        
        ####################################################################################
        # MNMA Rotation optimisation
        init_R = self.R_opt if self.use_multi else np.eye(3)
        R_torch, cov_torch = self.singleframe_optimiser.optimise(init_R, pred_norm_vec, pred_kappa_vec)
        self.R_opt = R_torch.detach().numpy()        # Optimised rotation estimate
        cov = cov_torch.detach().numpy()        # Covariance of rotation estimate

        ####################################################################################
        # Multiframe Optimisation
        if self.use_multi:
            self.R_opt = self.multiframe_optimiser.optimise(self.R_opt, cov)

        return self.R_opt.copy(), norm_out, kappa_out

    def preprocess_img(self, img, format):
        H, W, D = img.shape
        if format == 'BGR':
            assert(D==3)
            img_rgb = img[:, :, ::-1]
        elif format == 'Grayscale':
            assert(D==1)
            img_rgb = np.dstack((img, img, img))
        elif format == 'RGB':
            assert(D==3)
            img_rgb = img
        else:
            print('Unkown image format requested!')
            return None
        
        # img: tensor, RGB, (1, 3, H, W)
        img_torch = torch.from_numpy(img_rgb).to(self.device).permute(2, 0, 1).to(dtype=torch.float32)
        img_torch = (img_torch / 255.0).unsqueeze(0).contiguous()
        img_torch = self.normalise_fn(img_torch)

        return img_torch

    def create_visualisation(self, color_image, R_opt, norm_out, kappa_out, DISPLAY_MODE, show_mf=True, show_fps=True):
        img_vis = vis_utils.visualize_pred(color_image, norm_out, kappa_out, DISPLAY_MODE) 
        h, w = img_vis.shape[:2]
        if show_mf:
            img_vis = vis_utils.visualize_MFinImage(img_vis, R_opt, line_thickness=4)

        # Display title
        img_vis = cv2.rectangle(img_vis, (0, 0), (self.TITLE_WIDTHS[DISPLAY_MODE], 40), (0,0,0), -1)
        img_vis = cv2.putText(img_vis, DISPLAY_MODE, (10,30), self.TITLE_FONT,  
            1, (255,255,255), 2, cv2.LINE_AA) 
        
        if show_fps:
            new_frame_time = time.time()
            fps = int(1/(new_frame_time-self.prev_frame_time)) 
            self.prev_frame_time = new_frame_time
            cv2.putText(img_vis, str(fps)+' fps', (w-80, h-7), self.TITLE_FONT, 0.7, (100, 255, 0), 2, cv2.LINE_AA) 
        
        return img_vis