#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to invert spectra from MODIS and estimate snow cover and properties
for 2 x 7 band pixels from a scene

"""

import numpy as np
import h5py as h5
import scipy.interpolate as irp
from spires_core import speedyinvert

import matplotlib.pyplot as plt

#R is 7 band spectra for the 2 pixels
R=np.array([[0.8203,0.6796,0.8076,0.8361,0.1879,0.0321,0.0144],
           [0.4773,0.4482,0.4474,0.4823,0.1815,0.1019,0.0748]])

#R0 is the 7 band background spectra
R0=np.array([[0.2219,0.2681,0.1016,0.1787,0.3097,0.2997,0.2970],
            [0.1377,0.2185,0.0807,0.1127,0.2588,0.2696,0.1822]])

#modis central wavelengths, for plotting
wl=np.array([0.6450,0.8585,0.4690,0.5550,1.2400,1.6400,2.1300])

#need to sort those as the MODIS bands don't go in increasing order
idx=wl.argsort(axis=0)

#matlab solutions for: fractional snow cover(fsca), 
#fractional shade(fshade),
#grain size (um), and dust (ppm)
msol=np.array([[0.8930,0.1070,367.8819,20.1914],
              [0.4939,0.5061,83.5660,44.9953]])
#matlab solutions for reflectance corresponding to above
mrefl=np.array([[0.8143,0.6879,0.8194,0.8240,0.1865,0.0084,0.0040],
               [0.4680,0.4354,0.4654,0.4684,0.2309,0.0339,0.0175]])
#matlab RMSE (2 solutions x 2 days because the solver tries a mixed pixel
#(fsca,fshade,fother)
# and a fully snow covered pixel (fsca,fshade only), so 2 solutions for
# 2 days
mrmse=np.array([[0.0186,0.0326],[0.0136,0.1055]])
#solar zenith angle for both days
solarZ=np.array([24.0,24.71])

#look up table location
#the look up table was created doing radiative transfer Mie scattering calcs
Ffile='LUT_MODIS.mat'
#ideal shade endmember
shade=0

#load LUT
if 'F' not in locals():
    f=h5.File(Ffile,'r')
    d={}
    for k in f.keys():
        d[k]=np.squeeze(np.array(f.get(k)))
    f.close()
#create 4-D interpolant
#with the following structure
#reflectance = F(band # (1-7), solar zenith angle (0-90 deg), dust (0-1000 ppm),
#grain radius (30-1200 um)
#I've checked to make sure results are the same as MATLAB and they match
    F=irp.RegularGridInterpolator(
        points=[d['X4'],d['X3'],d['X2'],d['X1']],
        values=d['X'])        
    plt.close('all')

#run inversion and plot results for both pixels
for i in range(0,len(R)):
    #run inversion
    res=speedyinvert(F,R[i],R0[i],solarZ[i],shade)
    
    #plot results
    plt.figure(i)
    plt.plot(wl[idx],R[i][idx],'k-', label='R')
    plt.plot(wl[idx],res[1][idx],'r-', label=
             'python model:\n(RMSE=%0.2f,%0.2f)\nfsca=%0.2f\nfshade=%0.2f\ndust=%d ppm\nrg=%d ug' 
             %(res[2].fun,res[3].fun,res[0].x[0],res[0].x[1],res[0].x[2],res[0].x[3]))
    plt.plot(wl[idx],mrefl[i][idx],'g-', label=
             'matlab model:\n(RMSE=%0.2f,%0.2f*):\nfsca=%0.2f\nfshade=%0.2f\ndust=%d ppm\nrg=%d ug' 
             %(mrmse[i][0],mrmse[i][1],msol[i][0],msol[i][1],msol[i][3],msol[i][2]))
    plt.legend(loc="upper right")
    plt.title('pixel %d' %i)
    