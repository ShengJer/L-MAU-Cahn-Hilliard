import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifftshift, fft2, ifft2
import pickle as pk
from scipy.optimize import fsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable


def threshold(x, I_trial, vf):
    m1 = (I_trial >= x).astype(np.int64)
    m1[m1 == 0] = 1e-04
    N = I_trial.shape[0] * I_trial.shape[1] 
    I_next = I_trial*m1
    return np.sum(I_next) - vf*N


def GS_reconstruction(im_GT, # a certain ground truth microstructure
                   im_trial, # a initial trial microstructure
                   M_true, # the ground truth fourier amplitude
                   target_autoc, # the ground truth autocorrelation
                   tolerance=1e-08, 
                   max_iteration=3000,
                   print_out=50
                   ):
    N = im_trial.shape[0] * im_trial.shape[1]
    F_trial = fft2(im_trial)
    P_trial = np.absolute(F_trial)**2
    M_trial = np.sqrt(P_trial)
    autoc_trial = np.absolute(fftshift(ifft2(P_trial)))
    vf = im_trial.mean()
    for i in range(max_iteration):
        m = F_trial * (M_true/M_trial)
        I_trial = np.absolute(fftshift(ifft2(m))) # Fourier transform into real space
        
        F_flag = fft2(I_trial)
        P_flag = np.absolute(F_flag)**2
        autoc_flag = np.absolute(fftshift(ifft2(P_flag)))
        error = np.sum((autoc_flag/N  - target_autoc/N)**2)
        if i % print_out ==0:
            print('After {} run, square error = {}'.format(i, error))
        if error <= tolerance:
            print ("ther error is smaller than the tolerance: break")
            break
        
        v = np.linspace(0, 1, 300)
        g = np.zeros(v.shape) 
        for j in range(len(v)):
            g[j] = threshold(v[j], I_trial, vf=vf) 
        # to choose a theshold such that all value smaller than theshold becomes zeros while other retains the value
        ## the most important is that the total volume fraction should remain the same ==> g gives the threshold error for volume fraction        
        select = np.argmin(np.absolute(g))
        guess0 = v[select] # select the threshold that makes total microstructrure meets required volume fraction
        args = (I_trial, vf)
        av = fsolve(threshold, guess0, args)
        if i % print_out ==0:
            print('After %d run, threshold = %.10f' % (i, av))
            
        I_trial[I_trial < av] = 1e-04
        
        I_next = I_trial


        F_trial = fft2(I_next)
        P_trial = np.absolute(F_trial)**2
        M_trial = np.sqrt(P_trial)
    
    
    print("Finally, the restraint vf={:.5f}, the reconstruct vf={:.5f}".format(vf, I_trial.mean()))
    