#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:22:50 2022

@author: nbair
"""

import numpy as np
import h5py as h5
import scipy.interpolate as irp
from spires_core import speedyinvert
import matplotlib.pyplot as plt

R=np.array([[0.8203,0.6796,0.8076,0.8361,0.1879,0.0321,0.0144],
           [0.4773,0.4482,0.4474,0.4823,0.1815,0.1019,0.0748]]
           )
R0=np.array([[0.2219,0.2681,0.1016,0.1787,0.3097,0.2997,0.2970],
            [0.1377,0.2185,0.0807,0.1127,0.2588,0.2696,0.1822]])
#modis central wavelengths
wl=np.array([0.6450,0.8585,0.4690,0.5550,1.2400,1.6400,2.1300])
idx=wl.argsort(axis=0)

msol=np.array([[0.8930,0.1070,367.8819,20.1914],
              [0.4939,0.5061,83.5660,44.9953]])
mrefl=np.array([[0.8143,0.6879,0.8194,0.8240,0.1865,0.0084,0.0040],
               [0.4680,0.4354,0.4654,0.4684,0.2309,0.0339,0.0175]])
mrmse=np.array([[0.0186,0.0326],[0.0136,0.1055]])

solarZ=np.array([24.0,24.71])

Ffile='LUT_MODIS.mat'
shade=0

#load LUT
if 'F' not in locals():
    f=h5.File(Ffile,'r')
    d={}
    for k in f.keys():
        d[k]=np.squeeze(np.array(f.get(k)))
    f.close()
#create interpolant
    F=irp.RegularGridInterpolator(
        points=[d['X4'],d['X3'],d['X2'],d['X1']],
        values=d['X'])        
    plt.close('all')

for i in range(0,len(R)):
    #run inversion
    res=speedyinvert(F,R[i],R0[i],solarZ[i],shade)
    
    #plot results
    plt.figure(i)
    plt.plot(wl[idx],R[i][idx],'k-', label='R')
    plt.plot(wl[idx],res[1][idx],'r-', label=
             'python model: (RMSE=%0.2f,%0.2f) fsca=%0.2f rg=%d ug dust=%d ppm' 
             %(res[2].fun,res[3].fun,res[0].x[0],res[0].x[2],res[0].x[3]))
    plt.plot(wl[idx],mrefl[i][idx],'g-', label=
             'matlab model (RMSE=%0.2f,%0.2f-selected): fsca=%0.2f rg=%d ug dust=%d ppm' 
             %(mrmse[i][0],mrmse[i][1],msol[i][0],msol[i][2],msol[i][3]))
    plt.legend(loc="upper right")
    plt.title('pixel %d' %i)
    