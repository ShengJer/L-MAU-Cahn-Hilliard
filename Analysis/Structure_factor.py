import numpy as np
import os
from scipy.fft import fftn
from scipy.fft import fftshift, ifftshift
from scipy.fftpack import fftfreq


def structure_factor(array, hx, hy, nbins, idex):
    Nx = array.shape[1]
    Ny = array.shape[2]
    average= np.mean(array, axis=(1,2))
    array = array - average[:, None, None]

    p = fftshift(fftfreq(Nx, hx))

    q = fftshift(fftfreq(Ny, hy))
    kx = 2*np.pi* p
    ky = 2*np.pi* q
    
    kkx, kky = np.meshgrid(kx, ky)
    dis= np.sqrt(kkx**2 + kky**2)
    
    CL = []
    radial = []
    k1_store = []
    for i in idex:
        F = ifftshift(fftn(fftshift(array[i, :, :])))
        P = np.absolute(F**2)
        
        r_max = dis.max()
        bin_size = (r_max / nbins) # the interval of x axis
        bins = np.arange(bin_size, r_max, step=bin_size) # the x axis to rmax grid point
        radial_sum = np.zeros_like(bins)
        for j, r in enumerate(bins):
            # Generate Radial Mask from dt using bins
            mask = (dis <= r) * (dis > (r - bin_size)) # determine the location between [bin bin+bin_size]
            if np.sum(mask) == 0:
                radial_sum[j] = 0
            else:
                radial_sum[j] = np.sum(P[mask]) / np.sum(mask)
        k1 = (radial_sum*bins).sum()/radial_sum.sum()
        k1_store.append(k1)
        CL.append(2*np.pi / k1)
        radial.append(radial_sum)
    return bins, radial, CL, k1_store

## radial = structure factor at different time
## k1_store = characteristic length at different time
