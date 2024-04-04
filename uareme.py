""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)
"""
import argparse
import cv2
import torch
import numpy as np
import time
import yaml

import utils.input as input_utils
import utils.MFOpt as mfOpt_utils
from utils.MNMAoptimiser import MNMAoptimiser
import utils.visualisation as vis_utils

if __name__ == '__main__':
    # Initialise device
    device = torch.device('cuda:0')

    # Input arguments parser: choose input and whether to save the estimated rotations
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='webcam', type=str, help=r'webcam: generic webcam input, /path/to/video.mp4: video file (.mp4 or .avi), "/wildcard/pattern_*img.png": images with wildcard (glob), but must be in quotes')
    parser.add_argument('--save_trajectory', action='store_true')
    args = parser.parse_args()

    # Read in default parameters
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Set parameters
    height = config.get('height')
    width = config.get('width')
    loop = config.get('loop')
    use_multi = config.get('use_multi')
    window_length = config.get('window_length')
    interframe_sigma = config.get('interframe_sigma')
    robust = config.get('robust')
    use_kappa = config.get('use_kappa')
    kappa_threshold = config.get('kappa_threshold')
    show_viewer = config.get('show_viewer')
    show_fps = config.get('show_fps')
    show_mf = config.get('show_mf')
    if args.input == 'webcam' and not show_viewer:
        show_viewer = True
        print("!!! No viewer is not valid for webcam input - Showing viewer !!!")
    if loop and not show_viewer:
        show_viewer = True
        print("!!! No viewer is not valid for looping input - Showing viewer !!!")

    # Display constants
    PAUSE = False                                                                                   # Pause or play demo
    MODES_DICT = {1: 'RGB', 2: 'Normals', 3: 'Confidence'}                                          # List of displays
    DISPLAY_MODE = 'RGB'
    TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX                                                           # Display title font
    TITLE_WIDTHS = {MODES_DICT[i]: cv2.getTextSize(t, TITLE_FONT, 1, 2)[0][0] 
                    + 20 for i, t in MODES_DICT.items()}                                            # Display title width 

    ####################################################################################################
    # INITIALISATION
    ####################################################################################################
    # Define input source
    InputStream = input_utils.define_input(args.input, device, H=height, W=width)
    # Load surface normal prediction model
    model = input_utils.define_model(device)

    # Initialise rotation to identity
    R_opt = np.eye(3)
    R_traj = []

    # Initialise multiframe Optimiser if enabled
    if use_multi:
        multiframeOptimiser = mfOpt_utils.gtsam_rot(window_length=window_length, interframe_sigma=interframe_sigma, robust=robust)

    # Initialise Rotation optimiser
    MNMAopt = MNMAoptimiser(H=height, W=width, use_kappa=use_kappa)

    if show_viewer:
        # Display
        display = cv2.namedWindow('U-ARE-ME', flags=cv2.WINDOW_GUI_NORMAL)
        print(f"Display mode: {DISPLAY_MODE}")
        prev_frame_time = time.time()

    ####################################################################################################
    # MAIN LOOP #
    ####################################################################################################
    print("*** RUNNING ***")
    while True:
        if not PAUSE:
            ####################################################################################
            # Get input frame
            data_dict = InputStream.get_sample()
            color_image = data_dict['color_image']  # For visualisation
            img = data_dict['img']                  # Tensor, (normalised) for processing

            ####################################################################################
            # Prediction of surface normals and their confidence (kappa)
            with torch.no_grad():
                norm_out = model(img)[0]
                pred_norm = norm_out[:, :3, :, :]
                pred_kappa = norm_out[:, 3:, :, :]
                pred_kappa[pred_kappa > kappa_threshold] = kappa_threshold
                pred_norm_vec = pred_norm[0,...].view(3, -1)
                pred_kappa_vec = pred_kappa[0,...].view(1, -1)

            ####################################################################################
            # MNMA Rotation optimisation
            init_R = R_opt
            R_torch, cov_torch = MNMAopt.optimise(init_R, pred_norm_vec, pred_kappa_vec)
            R_opt = R_torch.detach().numpy()        # Optimised rotation estimate
            cov = cov_torch.detach().numpy()        # Covariance of rotation estimate

            ####################################################################################
            # Multiframe Optimisation
            if use_multi:
                R_opt = multiframeOptimiser.optimise(R_opt, cov)

            ####################################################################################
            # Post processing
            R_traj.append(R_opt)

            if show_viewer:
                # Display RGB, NORMALS, CONFIDENCE, and draw on coordinate frame from current rotation estimate
                img_vis = vis_utils.visualize_pred(color_image, pred_norm, pred_kappa, DISPLAY_MODE)
                if show_mf:
                    img_vis = vis_utils.visualize_MFinImage(img_vis, R_opt, line_thickness=4)

                # Display title
                img_vis = cv2.rectangle(img_vis, (0, 0), (TITLE_WIDTHS[DISPLAY_MODE], 40), (0,0,0), -1)
                img_vis = cv2.putText(img_vis, DISPLAY_MODE, (10,30), TITLE_FONT,  
                    1, (255,255,255), 2, cv2.LINE_AA) 
                
                if show_fps:
                    new_frame_time = time.time()
                    fps = int(1/(new_frame_time-prev_frame_time)) 
                    prev_frame_time = new_frame_time
                    cv2.putText(img_vis, str(fps), (width-40, height-7), TITLE_FONT, 0.7, (100, 255, 0), 2, cv2.LINE_AA) 
 
        ####################################################################################
        if show_viewer:
            # Output display and keypress handling
            cv2.imshow("U-ARE-ME", img_vis)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):                            # Quit
                break 
            elif keypress == ord(' '):                          # Spacebar
                PAUSE = not PAUSE
            elif keypress in [ord(str(a)) for a in range(1,len(MODES_DICT)+1)]: # Numbers [1,2,3]: cycle through displays
                DISPLAY_MODE = MODES_DICT[int(keypress) - int(ord('0'))]
                print(f"Display mode: {DISPLAY_MODE}")
            
            if cv2.getWindowProperty('U-ARE-ME',cv2.WND_PROP_VISIBLE) < 1:        
                break
        
        if InputStream.end and not loop:
            break

    if args.save_trajectory:
        R_traj = np.vstack(tuple(R_traj))
        np.savetxt("trajectory.txt", R_traj)
        print("Saved rotation trajectory to ./trajectory.txt")

    cv2.destroyAllWindows()
    print("--- U-ARE-ME successfully terminated ---")
    exit()
