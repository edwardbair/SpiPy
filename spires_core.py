#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize 

def speedyinvert(F,R,R0,solarZ,shade):    
    # invert snow spectra
    # inputs: 
    #     F - RT/Mie LUT, gridded interpolant LUT w/ F(band # (1-7), 
    #     solar zenith angle (0-90 deg), dust (0-1000 ppm),
    #     grain radius (30-1200 um)
    #     R - target spectra, array of len 7
    #     R0 - background spectra, array of len 7
    #     solarZ - solar zenith angle, deg, scalar
    #     shade - ideal shade endmember, scalar
    # output:
    #     res -  results from chosen solution, dict
    #     modelRefl - reflectance for chosen solution, array of len 7
    #     res1,res2 - mixed pixel (fsca,fshade,fother) vs snow only (fsca,fshade)
    #     solutions. One of those will be the same as res
    
    #tolerance
    ftl=1e-9
    #solver method
    mth='SLSQP'
    #mth specific options
    op={'disp': True, 'iprint': 100, 'maxiter': 1000, 'ftol': ftl}
    #bounds: fsca, fshade, dust, grain size
    bnds=np.array([[0,1],[0,1],[0,1000],[30,1200]])

    #initial guesses for fsca,fshade,dust, & grain size
    x0=[0.5,0.05,10,250] #scale to 0-1
    
    #model reflectance preallocation
    modelRefl=np.zeros(len(R))
    mode=[]
    
    #objective function
    def SnowDiff(x):
        nonlocal modelRefl,mode
        #calc the Euclidean norm of modeled and measured reflectance
        #input:
            #x - parameters: fsca, fshade, dust, grain size, array of len 4
            #nonlocal vars from parent function
            #modelRefl - modelRefl to be filled out, array of len 7        
            #mode -  4 or 3 int
            #4 variable solution (1-fsca,2-fshade,fother(1-fsca-fshade),3-grain size,4-dust)
            #mode - 3 variable solution (1-fsca,fshade (1-fsca),2-grain size,3-dust)
        #fill in modelRefl for each band for snow properties
        # ie if pixel were pure snow (no fshade, no fother)
       
        if mode==4:
            for i in range(0,len(R)):
                #x[2] and x[3] are dust and grain size
                pts=np.array([i+1,solarZ,x[2],x[3]])
                modelRefl[i]=F(pts)
            #now adjust model reflectance for a mixed pixel, with x[0] and x[1]
            #as fsca,fshade, and 1-x[0]-x[1] as fother
            modelRefl=x[0]*modelRefl+x[1]*shade+(1-(x[0]-x[1]))*R0
        if mode==3:
            for i in range(0,len(R)):
                #x[1] and x[2] are dust and grain size
                pts=np.array([i+1,solarZ,x[1],x[2]])
                modelRefl[i]=F(pts)
            modelRefl[i]=F(pts)
            modelRefl=x[0]*modelRefl+(1-x[0])*shade
          
        #Euclidean norm of measured - modeled reflectance
        diffR=la.norm(R-modelRefl)
        return diffR
            
    #construct the bounds in the form of constraints
    #inequality: constraint is => 0 
    cons = []
    #  1-(x[0]+x[1]) >= 0 <-> 1 >= x[0]+x[1]
    #  mixed pixel contraint: 1 >= fsca+fshade <-> 1 = fsca+fshade+(1-fsca)
    cons.append(
        {"type": "ineq", "fun": lambda x: 1-x[0]+x[1]}
        )
    #run minimization w/ 4 variables to solve
    mode=4
    res1 = minimize(SnowDiff,x0,constraints=cons,
                    method=mth,options=op, bounds=bnds)
    #store modeled refl
    modelRefl1=modelRefl
    #run minimization w/ 3 variables to solve, no constraint needed
    mode=3
    x0=[0.5,10,250]
    bnds=np.array([[0,1],[0,1000],[30,1200]])
    res2 = minimize(SnowDiff,x0,method=mth,options=op, bounds=bnds)
    #insert a zero in for consistency for x[2] (fother)
    res2.x=np.insert(res2.x,1,1-res2.x[0])
    modelRefl2=modelRefl
    #if fsca is within 2 pct, use 3 variable solution
    #error will be higher, but 4 parameter solution likely overfits
    if (abs(res1.x[0]-res2.x[0]) < 0.02):
        choice=3
        res=res2
        modelRefl=modelRefl2
    else:
        choice=4
        res=res1
        modelRefl=modelRefl1
    
    return res,modelRefl,res1,res2,choice