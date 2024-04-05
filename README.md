# U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
[Aalok Patwardhan](https://aalpatya.github.io/)\*, [Callum Rhodes](https://scholar.google.com/citations?user=aQSQwcUAAAAJ&hl=en&oi=ao)\*, [Gwangbin Bae](https://www.baegwangbin.com/), [Andrew J. Davison](https://www.doc.ic.ac.uk/~ajd/).
(\* indicates equal contribution.)

Dyson Robotics Lab, Imperial College London

This code accompanies the U-ARE-ME paper (2024).


### Project page: https://callum-rhodes.github.io/U-ARE-ME/

<p align="center">
  <img src="https://github.com/callum-rhodes/U-ARE-ME/blob/main/docs/img/logo/uareme.gif">
</p>

## Initial Setup
A suitable CUDA enabled graphics card is required to run the system 'out of the box'. At least 4GB of VRAM is required for running the normals model.

Python>=3.9 is required.

We recommend using `python venv` to set up your environment
Run the following:

Clone the repository
```shell
git clone https://github.com/callum-rhodes/U-ARE-ME.git
cd U-ARE-ME
```
Create a python virtual environment
```shell
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
. venv/bin/activate

# Install pytorch as per instructions at https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Install the rest of the requirements
pip3 install -r requirements.txt
```
We use a pretrained surface normal estimation network (visit [DSINE (CVPR 2024)](https://github.com/baegwangbin/DSINE) for more info). Download the model weights from [here](https://drive.google.com/file/d/170KNIcId99_FmrZw9UiJZEnBIlFPkbi6/view?usp=sharing). Once downloaded, create a 'checkpoints' directory in the main repository and paste the dsine_v00.pt file into the checkpoints folder e.g.
```shell
mkdir checkpoints
mv ~/Downloads/dsine_v00.pt checkpoints/
```

## Run demo
Input can be a video file, path to images, or webcam (default)
To save the rotation estimates add the --save_trajectory argument

You can edit further parameters in the config.yml file.

```shell
# Webcam input
python uareme.py
# Video file input
python uareme.py --input <myvideo.mp4>
# Images input (with wildcard pattern)
python uareme.py --input 'path/to/images/patterns/*_img.png' # Wildcard path must be in quotes

```

## Citation
If you find this code/work to be useful in your research, please consider citing the following:

*[U-ARE-ME](https://callum-rhodes.github.io/U-ARE-ME/):*
```
@misc{patwardhan2024uareme,
    title={U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments}, 
    author={Aalok Patwardhan and Callum Rhodes and Gwangbin Bae and Andrew J. Davison},
    year={2024},
    eprint={2403.15583},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
*[DSINE (CVPR 2024)](https://github.com/baegwangbin/DSINE):*
```
@inproceedings{bae2024dsine,
    title={Rethinking Inductive Biases for Surface Normal Estimation},
    author={Gwangbin Bae and Andrew J. Davison},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

### Acknowledgement
This research has been supported by the EPSRC Prosperity Partnership Award with Dyson Technology Ltd.
