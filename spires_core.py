#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:59:38 2022

@author: nbair
"""

import numpy as np
import numpy.linalg as la
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
    
    #if less than 2 pct diff, use model 2 (fsca+fshade only)
    if (abs(res1[0]-res2[0]) < 0.02):
        res=res2
        modelRefl=modelRefl2
    else:
        res=res1
        modelRefl=modelRefl1
        
    return res,modelRefl,res1,res2
