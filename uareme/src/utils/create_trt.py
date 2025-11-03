""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)

This file contains a script to compile the standard pytorch model to TensorRT format
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from torch2trt import torch2trt
import input as input_utils
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = 'checkpoints/dsine_v00_trt.pth'

# Load surface normal prediction model
print("Loading model...")
model = input_utils.define_model(device)
dummy_input = torch.ones((1, 3, 480, 640)).to(device)
# Convert to TensorRT feeding sample data as input
print("Converting to Tensor RT (this could take some time)...")
model_trt = torch2trt(model, [dummy_input], fp16_mode=False)
print("Saving Tensor RT model...")
torch.save(model_trt.state_dict(), save_path)

print("--- TRT model saved to " + save_path + " ---")
