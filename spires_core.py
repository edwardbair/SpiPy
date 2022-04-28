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
        
    mth='COBYLA'
    op={'disp': True}
    bnds=[[0,1],[0,1],[30,1200],[0,1000]]
    x0=[0.5,0.05,250,10]
    modelRefl=np.zeros(len(R))
    
    def SnowDiff(x,modelRefl):
        
        for i in range(0,len(R)):
            pts=np.array([i+1,solarZ,x[3],x[2]])
            modelRefl[i]= F(pts)
            
        modelRefl=x[0]*modelRefl+x[1]*shade+(1-(x[0]-x[1]))*R0    
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
    cons.append(
        {"type": "ineq", "fun": lambda x: (x[0]+x[1])-1}
        );
    
    res1 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                    constraints=cons,options=op)
    modelRefl1=modelRefl
    #run again w/ only snow and shade endmembers
    #by adding new constraint
    #1-(x[0]+x[1]) <= 0 <-> 1 <= x[0]+x[1] 
    cons.append(
        {"type": "ineq", "fun": lambda x: 1-(x[0]+x[1])}
        )
    res2 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                      constraints=cons,options=op)
    modelRefl2=modelRefl
    #if fsca is with 2 pct, use fsca & fshade only (equality) solution
    if (abs(res1.x[0]-res2.x[0]) < 0.02):
        res=res2
        modelRefl=modelRefl2
    else:
        res=res1
        modelRefl=modelRefl1
        
    return res,modelRefl,res1,res2
