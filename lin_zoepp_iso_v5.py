# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:10:35 2016

@author: GrinevskiyAS
"""

from __future__ import division
import numpy as np
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import linalg as la
import bruges as b
np.set_printoptions(precision=3)


#поддержка кариллицы
font = {'family': 'Open Sans', 'weight': 'normal', 'size':12}
rc('font', **font)



def create_X_Y(vp,vs,rho,s1, t):
    X=np.zeros((2,2), dtype=complex)
    Y=np.zeros((2,2), dtype=complex)
    
  
    s3p = np.sqrt(((1/vp**2) - s1**2).astype(complex)) 
    s3s = np.sqrt(((1/vs**2) - s1**2).astype(complex))
    G = 1 - 2 * vs**2 * s1**2
    
    X[0,0] = vp * s1
    X[0,1] = vs * s3s
    X[1,0] = -1 * rho * vp * G
    X[1,1] = 2 * rho * vs**3 * s1 * s3s
    
    Y[0,0] = -2 * rho * vp * vs**2 * s1 * s3p
    Y[0,1] = -1 * rho * vs * G
    Y[1,0] = vp * s3p
    Y[1,1] = -1 * vs * s1

    
    return X,Y
    


alpha1=3000.0
beta1=1500.0
rho1=2400.0
alpha2=4000.0
beta2=1400.0
rho2=2075.0



an=np.linspace(0,89,100)
theta=an*pi/180.0



T=np.zeros_like(theta)
R=np.zeros_like(theta)
Rpp1=np.zeros_like(theta)
Rpp2=np.zeros_like(theta)
Rpp3=np.zeros_like(theta)
Rpp4=np.zeros_like(theta)


#for i, th in enumerate(theta):
#    s1 = sin(th)/alpha1    
#    X,Y = create_X_Y(alpha1, beta1, rho1, s1, th)
#    X2,Y2 = create_X_Y(alpha2, beta2, rho2, s1, th)
#    
#    if not (any(np.isnan(Y2)) or ):
#        first_xy=np.dot(la.inv(X), X2)
#        second_xy=np.dot(la.inv(Y), Y2)
#        Rm=np.dot(first_xy - second_xy, la.inv(first_xy + second_xy))
#        Tm=2*la.inv(first_xy + second_xy)
#        Rpp1[i]=Rm[0,0]
#    
#    else:
#        print "method 1 not used"
#        pass
#
#for i, th in enumerate(theta):
#    s1 = sin(th)/alpha1    
#    X,Y = create_X_Y(alpha1, beta1, rho1, s1, th)
#    X2,Y2 = create_X_Y(alpha2, beta2, rho2, s1, th)
#    
#    I = np.eye(2)    
#    
#    part_slag=la.inv(X).dot(X2).dot(la.inv(Y2)).dot(Y)
#    
#    
#    Rm=la.inv(part_slag + I).dot(part_slag - I)
#    Tm=2*la.inv(Y2).dot(Y).dot(la.inv(part_slag + I))
#    Rpp2[i]=Rm[0,0]
#
for i, th in enumerate(theta):
    s1 = sin(th)/alpha1    
    X,Y = create_X_Y(alpha1, beta1, rho1, s1, th)
    X2,Y2 = create_X_Y(alpha2, beta2, rho2, s1, th)
    
    I = np.eye(2)    
    
    part_slag=la.inv(Y).dot(Y2).dot(la.inv(X2)).dot(X)
    
    
    Rm=la.inv(I + part_slag).dot(I - part_slag)
    Tm=2*la.inv(X2).dot(X).dot(la.inv(I + part_slag))
    Rpp3[i]=Rm[0,0]

#for i, th in enumerate(theta):
#    s1 = sin(th)/alpha1    
#    X,Y = create_X_Y(alpha1, beta1, rho1, s1, th)
#    X2,Y2 = create_X_Y(alpha2, beta2, rho2, s1, th)
#    
#    first_xy=np.dot(la.inv(X2), X)
#    second_xy=np.dot(la.inv(Y2), Y)
#    
#    
#    Rm=la.inv(second_xy + first_xy).dot(second_xy-first_xy)
#    Tm=2*first_xy.dot(la.inv(second_xy+first_xy)).dot(second_xy)
#    Rpp4[i]=Rm[0,0]





vp1=alpha1
vs1=beta1
vp2=alpha2
vs2=beta2

rc_ar = b.reflection.akirichards(vp1, vs1, rho1, vp2, vs2, rho2, an)
rc_z = b.reflection.zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, an)



fgr = plt.figure(figsize=[8,8],facecolor='w')
ax=fgr.add_subplot(111);    
#ax.plot(an,Rpp4,ls='none', marker='o', mfc='none', c='m',label='Sch-Pro_3')
#ax.plot(an,Rpp1,lw=3, label='Sch-Pro_1')
#ax.plot(an,Rpp2, lw=2, c='y',label='Sch-Pro_2')
ax.plot(an,Rpp3,ls='--', c='r',label='Sch-Pro_3')



ax.plot(an,rc_ar,label='Aki-Richards')
ax.plot(an,rc_z,label='Zoeppritz')

ax.legend()
#ax.set_ylim((0, 0.2))
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Amplitude')
ax.grid()

if (alpha1/alpha2 < 1):
    theta_crit=np.arcsin(alpha1/alpha2)
    ax.axvline(180*theta_crit/pi, ls='--',alpha=0.5, c=[0.5,0.5,0.5])