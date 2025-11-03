""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)

This file contains the GTSAM implementation for multiframe optimisation
"""
import gtsam
try:
    _ = gtsam.BatchFixedLagSmoother
    _ = gtsam.FixedLagSmootherKeyTimestampMap
except:
    # Older versions of Python don't have GTSAM>4.2, which need these.
    import gtsam_unstable
    gtsam.BatchFixedLagSmoother = gtsam_unstable.BatchFixedLagSmoother
    gtsam.FixedLagSmootherKeyTimestampMap = gtsam_unstable.FixedLagSmootherKeyTimestampMap

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

class gtsam_rot():
    """ gtsam
        sequential optimisation of multiple frames of rotation
    """
    def __init__(self, window_length=20, interframe_sigma=1.0, robust=True):
        #Window length of sliding window optimisation
        self.window_length = window_length

        #Will assume rotations between frames should be approx identity (smoothing)
        self.interframe_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([interframe_sigma, interframe_sigma, interframe_sigma], dtype=np.float64))

        #Whether to use robust factors on the measurements
        self.bRobust = robust
        #Whether to computer marginal variances 
        self.bComputeMarginals = False

        # optimiser params
        self.fixedlagsmoother = gtsam.BatchFixedLagSmoother(self.window_length)

        self.identity_rotation = gtsam.Rot3(np.identity(3, dtype=np.float64))
        initial_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1], dtype=np.float64))

        new_vals = gtsam.Values()
        new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        new_factors = gtsam.NonlinearFactorGraph()
        new_vals.insert(0, self.identity_rotation)
        new_timestamps.insert((0, 0.0))
        new_factors.add(gtsam.PriorFactorRot3(0, self.identity_rotation, initial_noise))

        self.fixedlagsmoother.update(new_factors, new_vals, new_timestamps)

        self.current_estimates = [np.identity(3, dtype=np.float64)]
        self.current_variances = [None]
        self.relative_rotation_to_previous_frame = [None]

        self.current_timestamp = 0.0

    def optimise(self, R, cov=np.eye((3))):
        
        #Add latest measurement
        if R is not None:
            #input must be float64
            R = R.astype(np.float64)
            R = ScipyRot.from_matrix(R).as_matrix() #enforces orthogonality
            meas = gtsam.Rot3(R)
            meas_noise = gtsam.noiseModel.Gaussian.Covariance(cov.astype(np.float64))
            if self.bRobust:
                #Create robust loss on measurement to effectively discard outliers
                meas_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(0.1), meas_noise)

            #Initialise latest frame with its current measured rotation
            self.current_estimate = R
        else:
            meas = None
            meas_noise = None

        #Create the prior factors (i.e. measurements)
        rot = meas
        noise = meas_noise

        if rot is not None:
            self.current_timestamp += 1
            new_vals = gtsam.Values()
            new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
            new_factors = gtsam.NonlinearFactorGraph()
            new_vals.insert(int(self.current_timestamp), rot)
            new_timestamps.insert((int(self.current_timestamp), self.current_timestamp))
            new_factors.add(gtsam.PriorFactorRot3(int(self.current_timestamp), rot, noise))
            new_factors.add(gtsam.BetweenFactorRot3(int(self.current_timestamp-1.0), int(self.current_timestamp), self.identity_rotation, self.interframe_noise))

            #Update smoother with new values
            self.fixedlagsmoother.update(new_factors, new_vals, new_timestamps)
            opt_output = self.fixedlagsmoother.calculateEstimate()

            self.current_estimates = []
            #Convert to numpy rotation and store locally
            for frame in range(np.max([0,int(self.current_timestamp-self.window_length)]), int(self.current_timestamp)):
                self.current_estimates.append(ScipyRot.from_matrix(opt_output.atRot3(frame).matrix()).as_matrix())

        return self.current_estimates[-1]



            


