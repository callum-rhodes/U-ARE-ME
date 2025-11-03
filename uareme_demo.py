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
import warnings
import os
import uareme.src.config as uareme_cfg
import uareme.src.utils.input as input_utils
import uareme.src.utils.visualisation as vis_utils
from uareme.src.uareme_cls import UAREME
from uareme.src.utils.output import OutputWriter

if __name__ == '__main__':
    # Initialise device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Input arguments parser: choose input and whether to save the estimated rotations
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='webcam', type=str, help=r'webcam: generic webcam input, /path/to/video.mp4: video file (.mp4 or .avi), "/wildcard/pattern_*img.png": images with wildcard (glob), but must be in quotes')
    parser.add_argument('--save_trajectory', action='store_true')
    parser.add_argument('--output', default=None, type=str, help='Output video path')
    parser.add_argument('--ckpt_dir', default=None, type=str, help='Checkpoints path')
    args = parser.parse_args()

    # Read in default parameters
    CFG_DIR = os.path.dirname(os.path.abspath(uareme_cfg.__file__))
    with open(os.path.join(CFG_DIR, 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)

    # Set parameters
    height = config.get('height')
    width = config.get('width')
    loop = config.get('loop')
    use_trt = config.get('use_trt')
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
        warnings.warn("CONFIG WARN: No viewer selected for webcam input!!!")
    if loop and not show_viewer:
        warnings.warn("CONFIG WARN: No viewer selected for looping input!!!")

    # Display constants
    PAUSE = False                                                                                   # Pause or play demo
    ITERATION = 0

    ####################################################################################################
    # INITIALISATION
    ####################################################################################################
    # Display title
    vis_utils.display_UAREME()
    # UAREME class
    uareme = UAREME(b_trt_model=use_trt, b_kappa=use_kappa, kappa_threshold=kappa_threshold,
                    b_multiframe=use_multi, b_robust=robust, window_length=window_length,
                    interframe_sigma=interframe_sigma, checkpoints_dir=args.ckpt_dir)
    # Define input source
    InputStream = input_utils.define_input(args.input, device, H=height, W=width)
    OutputWriter = OutputWriter(args.output, fps=30)
    R_traj = []

    print("------------------ U-ARE-ME started --------------------")
    if show_viewer:
        # Display
        display = cv2.namedWindow('U-ARE-ME', flags=cv2.WINDOW_GUI_NORMAL)
        print(f"Display mode: {uareme.DISPLAY_MODE}")

    ####################################################################################################
    # MAIN LOOP #
    ####################################################################################################
    while True:
        if not PAUSE:
            if (ITERATION//4) % 2 == 0:
                print("                    ooo RUNNING ooo", end="\r")
            else:
                print("                    --- RUNNING ---", end="\r")
            ####################################################################################
            # Get input frame
            data_dict = InputStream.get_sample()
            color_image = data_dict['color_image']  # For visualisation

            R_opt, norm_out, kappa_out = uareme.run(color_image, format='RGB')
            # Returned values: R_opt (3, 3), norm_out (H, W, 3), kappa_out (H, W, 1)

            ####################################################################################
            # Post processing
            R_traj.append(R_opt)

            if show_viewer or args.output is not None:
                img_vis = uareme.create_visualisation(color_image, R_opt, norm_out, kappa_out, uareme.DISPLAY_MODE)
                if args.output is not None:
                    OutputWriter.write(img_vis)

            ITERATION += 1
 
        ####################################################################################
        if show_viewer:
            # Output display and keypress handling
            cv2.imshow("U-ARE-ME", img_vis)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):                            # Quit
                break 
            elif keypress == ord(' '):                          # Spacebar
                PAUSE = not PAUSE
            elif keypress in [ord(str(a)) for a in range(1,len(uareme.MODES_DICT)+1)]: # Numbers [1,2,3]: cycle through displays
                uareme.DISPLAY_MODE = uareme.MODES_DICT[int(keypress) - int(ord('0'))]
                display_string = f"Display mode: {uareme.DISPLAY_MODE}"
                print(display_string+" "*(uareme.DISPLAY_WIDTH-len(display_string)))
            
            if cv2.getWindowProperty('U-ARE-ME',cv2.WND_PROP_VISIBLE) < 1:        
                break
        
        if InputStream.end and not loop:
            break

    OutputWriter.close()
    if args.save_trajectory:
        R_traj = np.vstack(tuple(R_traj))
        np.savetxt("trajectory.txt", R_traj)
        print("Saved rotation trajectory to ./trajectory.txt")

    cv2.destroyAllWindows()
    print("----------- U-ARE-ME successfully terminated -----------")
    print("--------------------------------------------------------")
    exit()
