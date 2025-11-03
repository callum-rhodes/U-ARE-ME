""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)

This file contains the input-related classes and functions
"""
import os
import cv2
import glob
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

####################################################################################
# Helper functions #
####################################################################################

def load_checkpoint(fpath, model):
    ''' Loads checkpoint for the given model '''

    ckpt = torch.load(fpath, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

def define_model(device, trt=False, checkpoints_dir=None):
    ''' Load model for surface normal prediction '''
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints') if checkpoints_dir is None else checkpoints_dir
    if trt:
        from torch2trt import TRTModule
        model = TRTModule()
        model.load_state_dict(torch.load(file_path+'dsine_v00_trt.pth'))
    else:
        from models.dsine.v00 import DSINE_v00 as DSINE
        model = DSINE().to(device)
        checkpoint = os.path.join(file_path, 'dsine_v00.pt')
        model = load_checkpoint(checkpoint, model)
        model.eval()

    return model

class dotdict(dict):
    """ dot.notation access to dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def define_input(input, device=0, H=480, W=640):
    ''' Select input type. Can be generic webcam, or video file, or wildcard-file pattern
        useful for image directories with images such as *_img.png '''
    if input == 'webcam':
        # Generic Webcam
        InputStream = InputWebcam(device=device, H=H, W=W)

    elif '.mp4' in input or '.avi' in input:
        # Video input
        InputStream = InputVideo(device=device, H=H, W=W, video_path=input)

    elif 'youtube.com' in input:
        # Youtube link
        video_id = input.split('watch?v=')[1]
        InputStream = InputYoutube(device=device, H=H, W=W, video_id=video_id)

    else:
        # File pattern can include wildcards for image directories
        # eg. '~/images/*img.png', or also a single image path.
        # For argparse, the file pattern must be in quotes ' '.
        InputStream = InputFilePattern(device=device, H=H, W=W, file_pattern=input) 
    
    return InputStream

def get_crop(orig_H, orig_W, H, W):
    ''' Crop input to the desired dimensions '''
    if orig_W / orig_H >= W / H:
        # width is long
        new_W = int(orig_H * (W / H))
        left = (orig_W - new_W) // 2
        right = left + new_W
        top = 0
        bottom = orig_H
    else:
        # height is long
        new_H = int(orig_W * (H / W))
        top = (orig_H - new_H) // 2
        bottom = top + new_H
        left = 0
        right = orig_W

    return top, bottom, left, right

####################################################################################
# Input functions #
####################################################################################
class InputWebcam():
    ''' Generic Webcam input '''
    def __init__(self, device=0, H=480, W=640):
        self.device = device
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            assert self.cap.isOpened()
        except:
            self.cap = cv2.VideoCapture(0)
            assert self.cap.isOpened()
        self.H = H
        self.W = W

        # Get crop dimensions
        _, frame = self.cap.read()
        orig_H, orig_W, _ = frame.shape
        self.top, self.bottom, self.left, self.right = get_crop(orig_H, orig_W, H, W)

        self.end = False

        # Normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        ''' Get sample from the Input Stream '''
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                color_image = frame
                break

        # color_image: BGR
        color_image = color_image[self.top:self.bottom, self.left:self.right, :]
        color_image = cv2.resize(color_image, (self.W, self.H))
        
        # img: tensor, RGB, (1, 3, H, W)
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(color_image).to(self.device).permute(2, 0, 1).to(dtype=torch.float32)
        img = (img / 255.0).unsqueeze(0).contiguous()
        img = self.normalize(img)

        # sample
        sample = {
            'color_image': color_image,   # RGB to BGR
            'img': img,
        }

        return sample


class InputVideo():
    def __init__(self, device=0, H=480, W=640, video_path=None):
        self.device = device
        self.video_path = video_path

        self.vidcap = cv2.VideoCapture(self.video_path)
        success, self.next_image = self.vidcap.read()
        orig_H, orig_W, _ = self.next_image.shape
        self.top, self.bottom, self.left, self.right = get_crop(orig_H, orig_W, H, W)
        self.H = H
        self.W = W

        self.end = False

        # normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):

        color_image = self.next_image
        
        # color_image: numpy, BGR, (H, W, 3)
        color_image = color_image[self.top:self.bottom, self.left:self.right, :]
        color_image = cv2.resize(color_image, (self.W, self.H))

        # img: tensor, RGB, (1, 3, H, W)
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(self.device).permute(2, 0, 1).to(dtype=torch.float32)
        img = (img / 255.0).unsqueeze(0).contiguous()
        img = self.normalize(img)

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
        }

        success, self.next_image = self.vidcap.read()
        if not success:
            self.end = True
            self.vidcap = cv2.VideoCapture(self.video_path)
            success, self.next_image = self.vidcap.read()

        return sample
    

class InputYoutube():
    def __init__(self, device=0, H=480, W=640, video_id='2y9-35c719Y'):
        from vidgear.gears import CamGear
        self.device = device
        self.url = 'https://www.youtu.be/' + video_id
        self.stream = CamGear(source=video_id, stream_mode=True, logging=False).start() # YouTube Video URL as input

        self.H = H
        self.W = W
        self.next_image = self.stream.read()
        orig_H, orig_W, _ = self.next_image.shape
        self.top, self.bottom, self.left, self.right = get_crop(orig_H, orig_W, H, W)

        self.end = False

        # normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        
        # color_image: numpy, BGR, (H, W, 3)
        color_image = self.next_image
        color_image = color_image[self.top:self.bottom, self.left:self.right, :]
        color_image = cv2.resize(color_image, (self.W, self.H))

        # img: tensor, RGB, (1, 3, H, W)
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(color_image).to(self.device).permute(2, 0, 1).to(dtype=torch.float32)
        img = (img / 255.0).unsqueeze(0).contiguous()
        img = self.normalize(img)

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
        }

        self.next_image = self.stream.read()
        if self.next_image is None:
            self.end=True

        return sample



class InputFilePattern():
    def __init__(self, device=0, 
                 H=480, W=640, file_pattern=None):
        self.device = device
        self.image_paths = [os.path.abspath(f) for f in glob.glob(file_pattern)]
        if not len(self.image_paths):
            raise FileExistsError("File not valid, or not found")
        self.image_paths.sort()

        self.H = H
        self.W = W

        self.end = False

        # normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.cur_idx = 0

    def get_sample(self):
        image_path = self.image_paths[self.cur_idx]

        # color_image: numpy, BGR, (H, W, 3)
        color_image = Image.open(image_path).convert("RGB")
        color_image = np.array(color_image)[:, :, ::-1]

        # crop and resize
        orig_H, orig_W, _ = color_image.shape
        top, bottom, left, right = get_crop(orig_H, orig_W, self.H, self.W)
        color_image = color_image[top:bottom, left:right, :]
        color_image = cv2.resize(color_image, (self.W, self.H))
        
        # img: tensor, RGB, (1, 3, H, W)
        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(self.device).permute(2, 0, 1).to(dtype=torch.float32)
        img = (img / 255.0).unsqueeze(0).contiguous()
        img = self.normalize(img)

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
        }

        self.cur_idx += 1
        if self.cur_idx >= len(self.image_paths):
            self.cur_idx = 0
            self.end = True
        return sample