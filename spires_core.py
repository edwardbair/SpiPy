#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:59:38 2022

@author: nbair
"""

import numpy as np
import numpy.linalg as la
<<<<<<< HEAD
import pyomo.environ as pyomo

def speedyinvert(F,R,R0,solarZ,shade):    
        
    model = pyomo.ConcreteModel()
    model.x1=pyomo.Var(initialize=0.5, bounds=(0,1))
    model.x2=pyomo.Var(initialize=0.05, bounds=(0,1))
    model.x3=pyomo.Var(initialize=250, bounds=(30,1200))
    model.x4=pyomo.Var(initialize=10, bounds=(0,1000))
    model.modelRefl=np.zeros(len(R))
    
    def SnowDiff(m):

        for i in range(0,len(R)):
            pts=np.array([i+1,solarZ,m.x4.value,m.x3.value])
            m.modelRefl[i]= F(pts)
            
        m.modelRefl=m.x1.value*m.modelRefl+m.x2.value*shade+(1-(m.x1.value-m.x2.value))*R0    
        diffR=la.norm(R-m.modelRefl)
        return diffR
=======
from scipy.optimize import minimize#,basinhopping

def speedyinvert(F,R,R0,solarZ,shade):    
    
    #sz=0.3
    jxa='3-point'
    tl=1e-9
    mth='SLSQP'
    op={'disp': True}
    bnds=np.array([[0,1],[0,1],[30,1200],[0,1000]])

    #x0=[0.5,0.05,250,10]
    x0=np.array([0.5,0.05,0.188,0.01])
    
    modelRefl=np.zeros(len(R))

    def SnowDiff(x,modelRefl):
        #prevent out of bounds guesses
        x[x>1]=1
        x[x<0]=0
        #re-scale from 0-1 to original values
        for i in range(0,len(x)):
            rng=bnds[i][1]-bnds[i][0]  
            x[i]=x[i]*rng+bnds[i][0]
    
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
        lower, upper = [0,1]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
>>>>>>> tmp
    
    model.obj = pyomo.Objective(rule=SnowDiff)
    model.c1 = pyomo.Constraint( expr = model.x1 + model.x2 <= 1)  
    opt = pyomo.SolverFactory('glpk')
    opt.solve(model)
    modelRefl1=model.modelRefl
    res1=[model.x1.value,model.x2.value,model.x3.value,model.x4.value]
    
    model.c1.deactivate
    model.c2 = pyomo.Constraint ( expr = model.x1 + model.x2 == 1)
    opt.solve(model)
    modelRefl2=model.modelRefl
    res2=[model.x1.value,model.x2.value,model.x3.value,model.x4.value]
    
<<<<<<< HEAD
    #if less than 2 pct diff, use model 2 (fsca+fshade only)
    if (abs(res1[0]-res2[0]) < 0.02):
=======
    res1 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                      constraints=cons,options=op,tol=tl, jac=jxa)
    # minimizer_kwargs = dict(args=modelRefl, method=mth, 
    #                         options=op, constraints=cons)
    # res1 = basinhopping(SnowDiff, x0, minimizer_kwargs=minimizer_kwargs,
    #                     stepsize=sz)
    
    modelRefl1=modelRefl
    #run again w/ only snow and shade endmembers
    #by adding new constraint
    #1-(x[0]+x[1]) <= 0 <-> 1 <= x[0]+x[1] 
    cons.append(
        {"type": "ineq", "fun": lambda x: 1-(x[0]+x[1])}
        )
    res2 = minimize(SnowDiff,x0,args=modelRefl,method=mth,
                      constraints=cons,options=op,tol=tl, jac=jxa)
    
    # res2 = basinhopping(SnowDiff, x0, minimizer_kwargs=minimizer_kwargs,
    #                     stepsize=sz)
    modelRefl2=modelRefl
    #if fsca is with 2 pct, use fsca & fshade only (equality) solution
    if (abs(res1.x[0]-res2.x[0]) < 0.02):
>>>>>>> tmp
        res=res2
        modelRefl=modelRefl2
    else:
        res=res1
        modelRefl=modelRefl1
        
    return res,modelRefl,res1,res2
