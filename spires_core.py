#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:59:38 2022

@author: nbair
"""

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize

def speedyinvert(F,R,R0,solarZ,shade):    
        
    mth='SLSQP'
    op={'disp': True,'maxiter':1000,'ftol':1e-9 }
    bnds=((0,1),(0,1),(30,1200),(0,1000))
    x0=(0.5,0.05,250,10)
    modelRefl=np.zeros(len(R))
    
    def SnowDiff(x,modelRefl):
        
        for i in range(0,len(R)):
            pts=np.array([i+1,solarZ,x[3],x[2]])
            modelRefl[i]= F(pts)
            
        modelRefl=x[0]*modelRefl+x[1]*shade+(1-(x[0]-x[1]))*R0    
        diffR=la.norm(R-modelRefl)
        return diffR
            
    cons= [{"type": "ineq", "fun": lambda x: np.array([1-x[0]+x[1]])}]
    res_ineq = minimize(SnowDiff,x0,args=modelRefl,method=mth,bounds=bnds,
                        constraints=cons,options=op)
    modelRefl_ineq=modelRefl
    #run again w/ only snow and shade endmembers
    cons= [
        {"type": "ineq", "fun": lambda x: np.array([1-x[0]+x[1]])},
        {"type": "eq", "fun": lambda x: np.array([1-x[0]+x[1]])}
        ]
    res_eq = minimize(SnowDiff,x0,args=modelRefl,method=mth,bounds=bnds,
                      constraints=cons,options=op)
    modelRefl_eq=modelRefl
    #if fsca is with 2 pct, use fsca & fshade only (equality) solution
    if (abs(res_eq.x[0]-res_ineq.x[0]) < 0.02):
        res=res_eq
        modelRefl=modelRefl_eq
    else:
        res=res_ineq
        modelRefl=modelRefl_ineq
        
    return res,modelRefl,res_ineq,res_eq
