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
#listed as 4 variable solution for pixel 1, then 3 variable solution for pixel 1,
#repeated for pixel 2
msol=np.array([[0.8848,0.0485,430.2819,18.2311], 
              [0.8929,0.1071,367.8753,20.1885],
              [0.4957,0.1262,487.8204,55.4392], 
              [0.4942,0.5058,83.3307,45.8701]])
#matlab solutions for reflectance corresponding to above
mrefl=np.array([
    [0.8173,0.6855,0.8166,0.8258,0.1848,0.0267,0.0230],
    [0.8143,0.6879,0.8194,0.8240,0.1865,0.0084,0.0040],
    [0.4828,0.4476,0.4524,0.4723,0.1814,0.1052,0.0706],
    [0.4681,0.4355,0.4653,0.4683,0.2307,0.0336,0.0174]])
#matlab RMSE (2 solutions x 2 days because the solver tries a mixed pixel
#(fsca,fshade,fother)
# and a fully snow covered pixel (fsca,fshade only), so 2 solutions for
# 2 days
mrmse=np.array([0.0186,0.0326,0.0136,0.1055])
#solar zenith angle for both days
solarZ=np.array([24.0,24.71])

#look up table location
#the look up table was created doing radiative transfer Mie scattering calcs
Ffile='LUT_MODIS.mat'
#ideal shade endmember
shade=np.zeros(len(R[0]))

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
for i in range(0, 2):
    #run inversion
    res = speedyinvert(F, R[i], R0[i], solarZ[i], shade)
    plt.figure(figsize=(10,5))
    #measured R
    plt.plot(wl[idx], R[i][idx], 'k-', label='R')
    #plot both 4 and 3 variable python solutions  
    str=('4-variable','3-variable')
    for j in range(0, 3, 2):        
        #python solution
        if j==0:
            str='4-variable'
        if j==2:
            str='3-variable'
        plt.plot(wl[idx], res[j+1][idx], 
                 label= str+' python model\n'+ '(RMSE=%0.2f) fsca=%0.2f fshade=%0.2f rg=%d ug dust=%d ppm'
             % (res[j].fun,res[j].x[0],res[j].x[1],res[j].x[2],res[j].x[3]))
    #plot both 4 and 3 variable matlab solutions
    for j in range(0,2):
        if j==0:
            str='4-variable'  
        if j==1:
            str='3-variable'    
        idt=2*(i+1)+(j)-2
        plt.plot(wl[idx], mrefl[idt][idx],
            label= str+' matlab model\n'+ '(RMSE=%0.2f) fsca=%0.2f fshade=%0.2f rg=%d ug dust=%d ppm'
            %(mrmse[idt],msol[idt][0],msol[idt][1],msol[idt][2],msol[idt][3]))

    plt.legend(loc="upper right")
    plt.title('pixel %d' %(i+1))
    plt.ylim(0,1)
    plt.xlabel('wavelength, um')
    plt.ylabel('reflectance')