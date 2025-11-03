""" 
U-ARE-ME: Uncertainty-Aware Rotation Estimation in Manhattan Environments
Aalok Patwardhan, Callum Rhodes, Gwangbin Bae, Andrew J. Davison 2024.
https://callum-rhodes.github.io/U-ARE-ME/
Copyright (c) 2024 by the authors.
This code is licensed (see LICENSE for details)

This file contains the rotation optimiser class
"""
import torch
from scipy.spatial.transform import Rotation as ScipyRot


class MNMAoptimiser:
    def __init__(self, use_kappa=True, max_iters=50):
            self.device = "cpu" #CPU is faster for small operations
            self.dtype = torch.float64
            self.use_kappa = use_kappa

            if self.use_kappa:
                self.S_func = self.S_kappa
            else:
                self.S_func = self.S_equal

            # Powers of x,y,z relating to the S coefficients in the cost function
            self.S_coeff_powers = [(4,0,0), (3,1,0), (2,2,0), (1,3,0), (0,4,0), (3,0,1), 
                                   (2,1,1), (1,2,1), (0,3,1), (2,0,2), (1,1,2), (0,2,2),
                                   (1,0,3), (0,1,3), (0,0,4)]

            # Helper matrices for getting the skew symmetric matrix from the SO(3) lie algebra vector
            self.W_row0 = torch.tensor([0,0,0,  0,0,1,  0,-1,0], dtype=self.dtype, device=self.device).view(3,3)
            self.W_row1 = torch.tensor([0,0,-1,  0,0,0,  1,0,0], dtype=self.dtype, device=self.device).view(3,3)
            self.W_row2 = torch.tensor([0,1,0,  -1,0,0,  0,0,0], dtype=self.dtype, device=self.device).view(3,3)

            #LM optimisation params
            self.max_iters = max_iters #Terminates at this step regardless
            self.ftol = 1e-6 #Amount of change in cost function below which will exit optimisation
            self.ptol = 1e-6 #Amount of change in rotation below which will exit optimisation
            self.gtol = 1e-6 #Amount of change in gradient below which will exit optimisation
            self.tau = 1e-3 #Initial scaling on the damping term (Higher is more damping)
            self.rho1 = .1 #Lower bound of rho value below which damping is increased
            self.rho2 = .75 #Upper bound of rho value above which damping is decreased
            self.gamma = 10 #Scalar on how much damping is decreased
            self.beta = 10 #Scalar on how much damping is increased

    def optimise(self, R_init, pred_norms, pred_kappa):
        R_init = torch.from_numpy(R_init).to(self.device).to(self.dtype).t()
        pred_norms = pred_norms.to(self.dtype)

        #First normalise normal vectors to make sure they are unit vectors
        pred_norms = torch.nn.functional.normalize(pred_norms, dim=0)
        
        #If we're applying kappa downweighting then process
        if self.use_kappa:
            pred_kappa = pred_kappa.to(self.dtype)
            pred_kappa /= pred_kappa.sum(dim=1) 

        #Go through the list of required S powers and calculated their related coefficients
        #This only needs to be done once per set of normals
        S_coeffs = torch.zeros((5,5,5), device=self.device, dtype=self.dtype)

        for u,v,w in self.S_coeff_powers:
                S_coeffs[u,v,w] = self.S_func(u, v, w, pred_norms, pred_kappa)

        #Calculated final S param S_{0,0,4}, based on the other params
        S_coeffs[0,0,4] = 1 - (S_coeffs[4,0,0] + S_coeffs[0,4,0] + 2*S_coeffs[2,2,0] + 2*S_coeffs[2,0,2] + 2*S_coeffs[0,2,2])

        #Get cost function matrix from the S coefficients
        M = self.get_M_matrix(S_coeffs)
  
        #Decompose into eigenvalues (lambda, L) and other bit (Q)
        #Since M is square symmetric, M = Q @ diag(L) @ Q^t, and can use eigh instead of eig
        L, Q = torch.linalg.eigh(M)

        L = abs(L)

        # M = H^t @ H therefore H = diag(L^0.5) @ Q^t
        self.H = torch.diag(torch.sqrt(L)) @ Q.t()

        #Pass precalculated values into the LM optimisation
        R_opt, covariance = self.LM_opt(R_init)
        R_opt = torch.tensor(ScipyRot.from_matrix(R_opt).as_matrix())

        return R_opt.t(), covariance


    def LM_opt(self, R_init):
        #Set current rotation based on initialisation conditions
        R_opt = R_init

        #Calculate intial total loss (sum of square errors, f(R)^t @ f(R)) and cost function values 
        loss, fR = self.cost_func(R_opt)

        #Calculate jacobian of d(f(R)) / d(delta(phi))
        J = self.jacobian_func(R_opt)
     
        #Get hessian matrix, J^t @ J   
        hess = J.t() @ J
        #Get gradient vector, J^t @ cost
        grad = J.t() @ fR

        #Set initial damping scaling
        u = self.tau * torch.max(hess.diag())

        #Optimisation loop
        for iter in range(self.max_iters-1):
            #Set damping based on current Hessian max value
            D = torch.eye(J.shape[1], device=J.device)
            D *= torch.max(torch.max(hess.diagonal(), D.diagonal()))

            #Scale damping
            damping = u*D
    
            #Solve Ax=b, (J^t@J + damping*I) = J^t@f(R)
            delta = -torch.linalg.solve((hess + damping), grad)
 
            #Calculate SO(3) delta rotation to apply to current estimate
            R_delta = self.so3exp(delta)
            R_step = R_delta @ R_opt

            #Get new cost values and loss for the new rotation 
            loss_step, f_step = self.cost_func(R_step)
 
            #Setting previous to current for the next iteration
            fR_prev = fR.clone()
            
            #Work out LM rho value to determine if we accept or reject this new rotation
            rho_denom = delta.t() @ (u*delta-grad)
            rho_nom = loss - loss_step
            rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf

            #If happy then set the current values to the new setpoint
            if rho > 0:
                R_opt = R_step
                J = self.jacobian_func(R_opt)
                grad = J.T @ f_step
                hess = J.T @ J
                loss = loss_step
                fR = f_step

            #Change damping scaling based on the rho value above
            u = u*self.beta if rho < self.rho1 else u/self.gamma if rho > self.rho2 else u
            torch.clamp(u, 1e-8, 1e8)

            # stop conditions
            gcon = max(abs(grad)) < self.gtol #Very small gradients (stationary point)
            pcon = (delta**2).sum()**0.5 < self.ptol #Very small changes to rotation (rotation is negligible)
            fcon = ((fR_prev-fR)**2).sum() < ((self.ftol*fR)**2).sum() if rho > 0 else False #+ve rho but small change in cost (cost stationary point)

            if gcon or pcon or fcon:
                break

        #recover covariance from optimal hessian
        cov = torch.linalg.inv(hess)
        
        return R_opt, cov
    

    def cost_func(self, R_opt):
        #Extract the r1,r2,r3 principal axes vectors of the MW frame in camera coords
        r1, r2, r3 = [R_opt[:,0].unsqueeze(1), R_opt[:,1].unsqueeze(1), R_opt[:,2].unsqueeze(1)]

        #Calculate V2(r_x) for each axis
        V2r1, V2r2, V2r3 = [self.get_V2r(r1), self.get_V2r(r2), self.get_V2r(r3)]

        #Calculate cost function
        fR = torch.vstack((self.H@V2r1,
                        self.H@V2r2,
                        self.H@V2r3))
        
        #Calculate loss, sum of squares of cost values
        eR = fR.t() @ fR

        return torch.log10(eR), fR
    
    
    def jacobian_func(self, R_opt):
            #Extract the r1,r2,r3 principal axes vectors of the MW frame in camera coords
            r1, r2, r3 = [R_opt[:,0].unsqueeze(1), R_opt[:,1].unsqueeze(1), R_opt[:,2].unsqueeze(1)]

            #Jacobian of dV2(r_x)/dr_x for each axis
            JVr1, JVr2, JVr3 = [self.get_Jac_V2r(r1), self.get_Jac_V2r(r2), self.get_Jac_V2r(r3)]

            #Skew symmetric matrices for each axis
            r1_skew, r2_skew, r3_skew = [self.skew_symmetric(r1), self.skew_symmetric(r2), self.skew_symmetric(r3)]

            #Jacobian of d(f(R)) / d(delta(phi))
            jac = -torch.vstack([self.H@JVr1@r1_skew,
                                self.H@JVr2@r2_skew,
                                self.H@JVr3@r3_skew])
            
            return jac

    
    @staticmethod
    def S_equal(u, v, w, norms, kappa):
        return (torch.mean((norms[0,:]**u * norms[1,:]**v * norms[2,:]**w))).detach().cpu()

    @staticmethod
    def S_kappa(u, v, w, norms, kappa): #Kappa is expected to be normalised weights i.e. sum(kappa)==1
        return (torch.sum(kappa * (norms[0,:]**u * norms[1,:]**v * norms[2,:]**w))).detach().cpu()


    @staticmethod
    def get_V2r(r):
        return torch.tensor([[r[0]**2], [r[1]**2], [r[2]**2], [r[0]*r[1]], [r[0]*r[2]], [r[1]*r[2]]], device=r.device, dtype=r.dtype)
    

    @staticmethod
    def get_Jac_V2r(r):
        return torch.tensor([[2*r[0],   0,      0],
                             [0,        2*r[1], 0],
                             [0,        0,      2*r[2]],
                             [r[1],     r[0],   0],
                             [r[2],     0,      r[0]],
                             [0,        r[2],   r[1]]],
                             device=r.device, dtype=r.dtype)
    

    def get_M_matrix(self, S):    
        M = torch.tensor([[S[2,2,0] + S[2,0,2],            -S[2,2,0],                  -S[2,0,2],                       S[1,3,0]-S[3,1,0]+S[1,1,2],                     S[1,0,3]-S[3,0,1]+S[1,2,1],                     -2*S[2,1,1]],
                            [-S[2,2,0],                     S[2,2,0]+S[0,2,2],          -S[0,2,2],                      S[3,1,0]-S[1,3,0]+S[1,1,2],                     -2*S[1,2,1],                                    S[0,1,3]-S[0,3,1]+S[2,1,1]],
                            [-S[2,0,2],                     -S[0,2,2],                  S[2,0,2]+S[0,2,2],              -2*S[1,1,2],                                    S[3,0,1]-S[1,0,3]+S[1,2,1],                     S[0,3,1]-S[0,1,3]+S[2,1,1]],                      
                            [S[1,3,0]-S[3,1,0]+S[1,1,2],    S[3,1,0]-S[1,3,0]+S[1,1,2], -2*S[1,1,2],                    S[2,0,2]+S[0,2,2]+S[4,0,0]-2*S[2,2,0]+S[0,4,0], S[0,1,3]+S[0,3,1]-3*S[2,1,1],                   S[1,0,3]+S[3,0,1]-3*S[1,2,1]],   
                            [S[1,0,3]-S[3,0,1]+S[1,2,1],    -2*S[1,2,1],                S[3,0,1]-S[1,0,3]+S[1,2,1],     S[0,1,3]+S[0,3,1]-3*S[2,1,1],                   S[2,2,0]+S[0,0,4]-2*S[2,0,2]+S[4,0,0]+S[0,2,2], S[1,3,0]+S[3,1,0]-3*S[1,1,2]],
                            [-2*S[2,1,1],                   S[0,1,3]-S[0,3,1]+S[2,1,1], S[0,3,1]-S[0,1,3]+S[2,1,1],     S[1,0,3]+S[3,0,1]-3*S[1,2,1],                   S[1,3,0]+S[3,1,0]-3*S[1,1,2],                   S[0,4,0]-2*S[0,2,2]+S[0,0,4]+S[2,2,0]+S[2,0,2]]],
                            device=self.device, dtype=self.dtype)
        
        return M
    

    def skew_symmetric(self, r):
        """ w1, w2, w3 = w[:]
            return torch.array([
                [  0, -w3,  w2], 
                [ w3,   0, -w1],
                [-w2,  w1,   0]
            ])
        """
        r = r.squeeze()
        return torch.stack([r @ self.W_row0.t(), r @ self.W_row1.t(), r @ self.W_row2.t()] , dim = -1)
    

    @staticmethod
    def so3log(R):
        """
            Maps SO(3) --> so(3) group. Holds for d between -1 and 1
        """
        if R[0,0] == R[1,1] == R[2,2] == 1.0:
            return torch.zeros(3, device=R.device, dtype=R.dtype)
        else:
            d = 0.5 * (torch.trace(R) - 1)

            lnR = (torch.acos(d) / (2 * torch.sqrt(1 - d**2))) * (R - R.T)

            w = torch.tensor([lnR[2, 1], lnR[0, 2], lnR[1, 0]], device=R.device, dtype=R.dtype)

        return w
    

    def so3exp(self, w):
        """
            Maps so(3) --> SO(3) group with closed form expression.
        """
        w.squeeze()
        theta = torch.linalg.norm(w)
        if theta < 1e-8:
            return torch.eye(3, device=w.device, dtype=w.dtype)
        else:
            w_hat = self.skew_symmetric(w)
            R = torch.eye(3, device=w.device, dtype=w.dtype) + (torch.sin(theta) / theta) * w_hat + ((1 - torch.cos(theta)) / theta**2) * (w_hat @ w_hat)
            return R


