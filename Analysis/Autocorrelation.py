import numpy as np
import os
from scipy.fft import fftn, ifftn
from scipy.fft import fftshift, ifftshift


def Autocorrelation(array, hx, hy, nbins):
    Nx = array.shape[0]
    Ny = array.shape[1]
    N= Nx*Ny

    F = ifftshift(fftn(fftshift(array)))
    P = np.absolute(F**2)
    autoc = np.absolute(ifftshift(ifftn(fftshift(P))))
    
    adj = np.reshape(autoc.shape, [2, 1, 1])
    inds = np.indices(autoc.shape) - adj / 2 # define grid point
    dt = np.sqrt(inds[0]**2 + inds[1]**2) # define each distance from origin to grid point
    
    r_max = int(Nx/2)
    bin_size = int(np.ceil(r_max / nbins)) # the interval of x axis
    bins = np.arange(bin_size, r_max, step=bin_size) # the x axis to rmax grid point
    radial_sum = np.zeros_like(bins)
    for i, r in enumerate(bins):
        # Generate Radial Mask from dt using bins
        mask = (dt <= r) * (dt > (r - bin_size)) # determine the location between [bin bin+bin_size]
        if np.sum(mask) == 0:
            radial_sum[i] = 0
        else:
            radial_sum[i] = np.sum(autoc[mask]) / np.sum(mask)
        # Return normalized bin and radially summed autoc
        norm_autoc_radial = radial_sum / np.max(autoc)
       
    return bins, norm_autoc_radial