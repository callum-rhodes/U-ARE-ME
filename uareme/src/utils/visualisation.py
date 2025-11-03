""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)

This file contains the visualisation classes and functions
"""
import numpy as np
import torch
import cv2
from matplotlib import cm

def display_UAREME():
    ''' Displays title when code is run '''
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print(r" _   _          ___  ______ ____        ___  ___ ____  ")
    print(r"| | | |        / _ \ | ___ \  __|       |  \/  ||  __| ")
    print(r"| | | |  ___  / /_\ \| |_/ / |__   ___  | .  . || |__  ")
    print(r"| | | | |___| |  _  ||    /|  __| |___| | |\/| ||  __| ")
    print(r"| |_| |       | | | || |\ \| |___       | |  | || |___ ")
    print(r" \___/        \_| |_/\_| \_\____/       \_|  |_/\____/ ")
    print("")
    print("--------------------------------------------------------")    

def tensor_to_numpy(tensor_in):
    ''' Pytorch tensor to numpy array '''
    if tensor_in is not None:
        if len(tensor_in.size()) == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif len(tensor_in.size()) == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception('invalid tensor size')
    return tensor_in

def normal_to_rgb(normal, normal_mask=None, to_numpy=True):
    ''' Converts the surfance normals into an RGB image '''
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    return normal_rgb

# depth to rgb
def depth_to_rgb(depth, d_min=None, d_max=None):
    ''' Normalises and clips depth data and produces a 3-channel RGB output '''
    depth = tensor_to_numpy(depth) if torch.is_tensor(depth) else depth

    if d_min is not None:
        depth[depth < d_min] = d_min
    else:
        d_min = np.min(depth)

    if d_max is not None:
        depth[depth > d_max] = d_max
    else:
        d_max = np.max(depth)
    
    depth = (depth - d_min) / abs(d_max - d_min)
    depth = (cm.gray(depth[:,:,0]) * 255).astype(np.uint8)[:,:,:3]
    return depth   


def visualize_pred(color_image, pred_norm, pred_kappa, mode='RGB'):
    ''' Display the desired output (RGB, Normals, Confidence)'''
    if mode=='Normals':
        normals = normal_to_rgb(pred_norm)
        return normals.copy() if len(pred_norm.shape) == 3 else normals[0,:,:,::-1].copy()
    elif mode=='Confidence':
        # Re-use the depth_to_rgb function for the confidence map
        pred_kappa = pred_kappa if len(pred_kappa.shape) == 3 else pred_kappa[0,...]
        return depth_to_rgb(pred_kappa, d_min=0.0)[...,::-1].copy()
    else:
        return color_image


def visualize_MFinImage(img, R_wc, line_length=100, line_thickness=2, center=None):
    ''' Draw a the current rotation estimate as a coordinate frame on the given image.
        UAREME provides the cam2world rotation estimate, R_wc, here we display the world2cam rotation estimate, R_cw.'''
    R_cw = R_wc.T
    x_line = (R_cw[:,0] * line_length).astype(int)
    y_line = (R_cw[:,1] * line_length).astype(int)
    z_line = (R_cw[:,2] * line_length).astype(int)

    center = (np.array(img.shape[0:2])/2).astype(int) if center is None else center

    lines = [x_line, y_line, z_line]
    colours = [(0,0,255), (0,255,0), (255,0,0)]

    z_order = np.flip(np.argsort([x_line[2], y_line[2], z_line[2]]))
    lines = [lines[z_order[0]], lines[z_order[1]], lines[z_order[2]]]
    colours = [colours[z_order[0]], colours[z_order[1]], colours[z_order[2]]]

    cv2.line(img, (center[1], center[0]), (center[1] + lines[0][0], center[0] + lines[0][1]), colours[0], thickness=line_thickness)
    cv2.line(img, (center[1], center[0]), (center[1] + lines[1][0], center[0] + lines[1][1]), colours[1], thickness=line_thickness)
    cv2.line(img, (center[1], center[0]), (center[1] + lines[2][0], center[0] + lines[2][1]), colours[2], thickness=line_thickness)


    return img