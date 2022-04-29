#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:59:38 2022

@author: nbair
"""

#note that basin hopping is commented out. That's a package to ensure results
#aren't stuck in a local minimum, which doesn't seem to help
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize#,basinhopping

def speedyinvert(F,R,R0,solarZ,shade):    
    #invert snow spectra
    #inputs: 
        #F - RT/Mie LUT, gridded interpolant LUT w/ F(band # (1-7), 
        #solar zenith angle (0-90 deg), dust (0-1000 ppm),
        #grain radius (30-1200 um)
        # R - target spectra, array of len 7
        # R0 - background spectra, array of len 7
        # solarZ - solar zenith angle, deg, scalar
        # shade - ideal shade endmember, scalar
    #output:
        #res -  results from chosen solution, dict
        #modelRefl - reflectance for chosen solution, array of len 7
        #res1,res2 - mixed pixel (fsca,fshade,fother) vs snow only (fsca,fshade)
        #solutions. One of those will be the same as res
    
    
    #params for optimization, have tweaked these alot
    #sz=0.3 # basin hopper param
    #jacobian
    jxa=None
    #tolerance
    tl=1e-9
    #solver method
    mth='SLSQP'
    #mth specific options
    op={'disp': True, 'iprint': 100, 'ftol': 1e-9, 'maxiter': 1000,
        'finite_diff_rel_step': None}
    #bounds: fsca, fshade, dust, grain size
    bnds=np.array([[0,1],[0,1],[0,1000],[30,1200]])

    #initial guesses for fsca,fshade,dust, & grain size
    #scaled 0-1, although that doesn't seem to impact results
    x0=[0.5,0.05,10,250] #scale to 0-1
    #x0=np.array([0.5,0.05,0.01,0.188])
    
    #model reflectance preallocation
    modelRefl=np.zeros(len(R))

    #objective function
    def SnowDiff(x,modelRefl):
        #calc the Euclidean norm of modeled and measured reflectance
        #input:
            #x - parameters: fsca, fshade, dust, grain size, array of len 4
            #modelRefl - modelRefl to be filled out, array of len 7
        #prevent out of bounds guesses
        for i in range(0,len(x)):
            if x[i]<bnds[i][0]:
                x[i]=bnds[i][0]
            if x[i]>bnds[i][1]:
                x[i]=bnds[i][1]
        
        
        #x[x>1]=1
        #x[x<0]=0
        #re-scale from 0-1 to original values
        # for i in range(0,len(x)):
        #     rng=bnds[i][1]-bnds[i][0]  
        #     x[i]=x[i]*rng+bnds[i][0]
            
        #fill in modelRefl for each band for snow properties
        # ie if pixel were pure snow (no fshade, no fother)
        #x[2] and x[3] are dust and grain size
        for i in range(0,len(R)):
            pts=np.array([i+1,solarZ,x[2],x[3]])
            modelRefl[i]= F(pts)
        #now adjust model reflectance for a mixed pixel, with x[0] and x[1]
        #as fsca,fshade, and 1-x[0]-x[1] as fother
        modelRefl=x[0]*modelRefl+x[1]*shade+(1-(x[0]-x[1]))*R0
        #Euclidean norm of measured - modeled reflectance
        diffR=la.norm(R-modelRefl)
        return diffR
            
    #construct the bounds in the form of constraints
    #inequality: constraint is =>0 vs. equality: constraint = 0
    #COBLYA only supports inequality constraints and not bounds
    #so bounds have to be set as inequality constraints
    cons = []
    for factor in range(len(bnds)):
        lower, upper = bnds[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    
    #  (x[0]+x[1])-1 <= 0 <-> x[0]+x[1] <= 1
    #  mixed pixel contraint: fsca+fshade+fother = 1
    cons.append(
        {"type": "ineq", "fun": lambda x: (x[0]+x[1])-1}
        );
    
    #run minimization
    res1 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                      constraints=cons,options=op,tol=tl, jac=jxa)
    # minimizer_kwargs = dict(args=modelRefl, method=mth, 
    #                         options=op, constraints=cons)
    # res1 = basinhopping(SnowDiff, x0, minimizer_kwargs=minimizer_kwargs,
    #                     stepsize=sz)
    
    #store modeled refl
    modelRefl1=modelRefl
    #run again w/ only snow and shade endmembers
    #by adding new constraint
    #1-(x[0]+x[1]) <= 0 <-> 1 <= x[0]+x[1] 
    #ie.1 <= fsca + fshade <= 1
    cons.append(
        {"type": "ineq", "fun": lambda x: 1-(x[0]+x[1])}
        )
    res2 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                      constraints=cons,options=op,tol=tl, jac=jxa)
    
    # res2 = basinhopping(SnowDiff, x0, minimizer_kwargs=minimizer_kwargs,
    #                     stepsize=sz)
    modelRefl2=modelRefl
    #if fsca is within 2 pct, use fsca & fshade only solution
    #error will be higher b/c of 3 params instead of 4. Idea is that
    # 4 parameter solution would overfit in this case
    if (abs(res1.x[0]-res2.x[0]) < 0.02):
        res=res2
        modelRefl=modelRefl2
    else:
        res=res1
        modelRefl=modelRefl1
    
    return res,modelRefl,res1,res2
