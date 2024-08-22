""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
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


class UAREME():
    def __init__(self,
                 img : np.ndarray,
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

        H, W, _ = img.shape
        self.singleframe_optimiser = MNMAoptimiser(H=H, W=W, use_kappa=self.use_kappa)

    def run(self, img : np.ndarray, format: str = 'RGB'):
        # Preprocess image to torch format
        img_torch = self.preprocess_img(img, format)

        ####################################################################################
        # Run image through network
        if img_torch is not None:
            with torch.no_grad():
                model_out = self.model(img_torch)[0]
                pred_norm = model_out[:, :3, :, :]
                norm_out = pred_norm.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
                pred_kappa = model_out[:, 3:, :, :]
                kappa_out = pred_kappa.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
                pred_kappa[pred_kappa > self.kappa_thresh] = self.kappa_thresh
                pred_norm_vec = pred_norm[0,...].view(3, -1)
                pred_kappa_vec = pred_kappa[0,...].view(1, -1)
        
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
        elif format == 'Greyscale':
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