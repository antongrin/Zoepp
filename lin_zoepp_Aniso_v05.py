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
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

import numpy
from mpl_toolkits.mplot3d import proj3d
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,-0.0001,zback]])
proj3d.persp_transformation = orthogonal_proj

from scipy import linalg as la
np.set_printoptions(precision=3)


#поддержка кариллицы
font = {'family': 'Open Sans', 'weight': 'normal', 'size':12}
rc('font', **font)

def create_X_Y(vp,vs,rho,s1, t):
    X=np.zeros((2,2), dtype=complex)
    Y=np.zeros((2,2), dtype=complex)
    
    E2 = (c11-c55)*(c33-c55) - (c13 + c55)**2
  
    X[0,0] = ep1
    X[0,1] = es1
    X[1,0] = -(c13*s1*ep1 + c33*s3p*ep3)
    X[1,1] = -(c13*s1*es1 + c33*s3s*es3)
    
    Y[0,0] = -c55*(s1*ep3 + s3p*ep1)
    Y[0,1] = -c55*(s1*es3 + s3s*es1)
    Y[1,0] = ep3
    Y[1,1] = es3

#    
#    mat=np.zeros((2,2), dtype=complex)   
#    mat[0,0] = c11*s1**2 + c55*s3**2 - rho
#    mat[0,1] = (c55 + c13)*s1*s3
#    mat[1,0] = (c55 + c13)*s1*s3
#    mat[1,1] = c55*s1**2 + c33*s3**2 - rho
#    
    
    return X,Y

  
def L_matrix(c, rho, fi, th):
    """Christoffel matrix G multiplied by V2/rho, for monoclinic media """

    a = c/rho
    n1 = cos(fi)*sin(th)
    n2 = sin(fi)*sin(th)
    n3 = cos(th)
    
    n = np.array((n1,n2,n3))
    
    
    a11=a[0,0]
    a12=a[0,1]
    a13=a[0,2]
    a16=a[0,5]
    a22=a[1,1]
    a23=a[1,2]
    a26=a[1,5]
    a33=a[2,2]    
    a36=a[2,5]
    a44=a[3,3]
    a45=a[3,4]
    a55=a[4,4]
    a66=a[5,5]
    


#Первый способ расчёта матрицы - по формулам из Цванкина для ORTH    
#    L=np.zeros((3,3))*np.nan    
#    L11 = a11*n1**2 + a66*n2**2 + a55*n3**2 + 2*a16*n1*n2
#    L22 = a66*n1**2 + a22*n2**2 + a44*n3**2 + 2*a26*n1*n2
#    L33 = a55*n1**2 + a44*n2**2 + a33*n3**2 + 2*a45*n1*n2
#    
#    L12 = a16*n1**2 + a26*n2**2 + a45*n3**2 +(a12+a66)*n1*n2
#    L13 = (a13 + a55)*n1*n3 + (a36 + a45)*n2*n3
#    L23 = (a36 + a45)*n1*n3 + (a23 + a44)*n2*n2    
      
#    L = np.array([[L11,L12,L13],
#                  [L12,L22,L23],
#                  [L13,L23,L33]])


#General matrix formulations
    L_cycle=np.zeros((3,3))*np.nan
    s=0
    for i in np.arange(0,3):
        for k in np.arange(0,3):
            s=0
            for j in np.arange(0,3):
                for l in np.arange(0,3):
                    if (k == l):
                        p2 = k
                    else:
                        p2 = 6 - (k+l)
                    if (i == j):
                        p1 = i
                    else:
                        p1 = 6 - (i+j)
                    
                    s = s+a[p1,p2]*n[j]*n[l]
                    
            L_cycle[i,k] = s
    

    return L_cycle

#def G_matrix(c,rho, )
def count_v_Christoffel(G):
    """ Tsvankin's monograph, appendix 1A, accepts Lambda matrix """
    
    G11=G[0,0]
    G22=G[1,1]
    G33=G[2,2]
    G12=G[0,1]
    G13=G[0,2]
    G23=G[1,2]
    
    
    
    a = -(G11 + G22 + G33)
    b = G11*G22 + G11*G33 + G22*G33 - G12**2 - G13**2 - G23**2
    c = G11*G23**2 + G22*G13**2 + G33*G12**2 - G11*G22*G33 - 2*G12*G13*G23
    
    d = -(a**2)/3 + b
    
    q = 2*(a/3)**3 - a*b/3 + c    
    Q = (d/3)**3 + (q/2)**2

#    if (Q<=0):
#        print 'Q <= 0, roots are real' 
#    else:
#        print 'roots are complex'

    nu = np.arccos(-q/(2*np.sqrt((-d/3)**3)))
    
    k=np.array((0,1,2))
    
    y = 2*np.sqrt(-d/3)*cos(nu/3 + k*2*pi/3)
    
    V = np.sqrt(y-a/3)
    
    return V




def cij_vti(vp0, vs0, rho, de, ep, ga):
    c33 = rho*vp0**2
    c44 = rho*vs0**2
    c11 = c33*(1 + 2*ep)
    c13 = np.sqrt((c33-c44)*(c33*(1+2*de)-c44))-c44
    c66 = c44*(1 + 2*ga)
    c12 = c11 - 2*c66
    
    c = np.array([[c11, c12, c13,   0,   0,   0],
                  [c12, c11, c13,   0,   0,   0],
                  [c13, c13, c33,   0,   0,   0],
                  [  0,   0,   0, c44,   0,   0],
                  [  0,   0,   0,   0, c44,   0],
                  [  0,   0,   0,   0,   0, c66]])
    return c
    

#def cij_hti(vp, vs, rho, dev, epv, gav):
#    c33 = rho*vp**2
#    c44 = rho*vs**2
#    c11 = c33*(1 + 2*ep)
#    c13 = np.sqrt((c33-c44)*(c33*(1+2*de)-c44))-c44
#    c66 = c44*(1 + 2*ga)
#    c12 = c11 - 2*c66
#    
#    c = np.array([[c11, c12, c13,   0,   0,   0],
#                  [c12, c11, c13,   0,   0,   0],
#                  [c13, c13, c33,   0,   0,   0],
#                  [  0,   0,   0, c44,   0,   0],
#                  [  0,   0,   0,   0, c44,   0],
#                  [  0,   0,   0,   0,   0, c66]])
#    return c
    
    
def cij_ort(vp0, vs0, rho, de1, ep1, ga1, de2, ep2, ga2, de3):
    c33 = rho*vp0**2
    c55 = rho*vs0**2
    
    c66 = c55*(1 + 2*ga1)
    c44 = c66/(1 + 2*ga2)    
    c22 = c33*(1 + 2*ep1) 
    c11 = c33*(1 + 2*ep2)    
 
    c13 = np.sqrt(2*de2*c33*(c33-c55) + (c33-c55)**2) - c55

    c23 = np.sqrt(2*de1*c33*(c33-c44) + (c33-c44)**2) - c44
    
    c12 = np.sqrt(2*de3*c11*(c11-c66) + (c11-c66)**2) - c66
    
    
    c = np.array([[c11, c12, c13,   0,   0,   0],
                  [c12, c22, c23,   0,   0,   0],
                  [c13, c23, c33,   0,   0,   0],
                  [  0,   0,   0, c44,   0,   0],
                  [  0,   0,   0,   0, c55,   0],
                  [  0,   0,   0,   0,   0, c66]])
    return c
    
    
        
    
        
    
        
    
    


def axisEqual3D(ax):
    """http://stackoverflow.com/a/9349255 """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
                  

def get_cube_points(a):
    x = np.array([-a,a,a,-a, -a,a,a,-a])
    y = np.array([-a,-a,a,a, -a,-a,a,a])
    z = np.array([-a,-a,-a,-a, a,a,a,a])
    return x,y,z



def fancy_aniso_plot(th, fi, vlist):
    
    maxlim = np.max(vlist)*1.1    
    
    anm,azm=np.meshgrid(th, fi)
    
    n1 = cos(azm)*sin(anm)
    n2 = sin(azm)*sin(anm)
    n3 = cos(anm)
    #
    
    f3d = plt.figure(facecolor='w')
    a3d = f3d.add_axes([0.1,0.1,0.8,0.8],projection='3d')
    
    a3d.scatter(vlist*n1, vlist*n2, vlist*n3, c = ((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))),
            cmap='Spectral_r', alpha=0.8, marker='o', s = 10, edgecolors = 'none', linewidths=0)


    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    

    xc,yc,zc = get_cube_points(maxlim)
    ax3.scatter(xc, yc, zc, alpha=0)
    ax3.scatter(0,0,0, alpha=0.7, s=1)
    ax3.set_xlim([-maxlim, maxlim])
    ax3.set_ylim([-maxlim, maxlim])
    ax3.set_zlim([-maxlim, maxlim])
    
    ax3.set_aspect('equal')
    
    
    f2d = plt.figure(facecolor='w', figsize=[12,4])
    a2d1 = f2d.add_subplot('131', aspect=1)
    a2d2 = f2d.add_subplot('132', aspect=1)
    a2d3 = f2d.add_subplot('133', aspect=1)
    
    
#    {x1, x3}   orth to x2 (Az = 90, fi = 0)
    ind_1 = (azm == np.radians(0))
    v_1 = vlist[ind_1]
    n1_1 = n1[ind_1]
    n3_1 = n3[ind_1]
    xdata_1 = np.hstack((-np.ravel(n1_1*v_1)[::-1], np.ravel(n1_1*v_1)))
    zdata_1 = -1 * np.hstack((np.ravel(n3_1*v_1)[::-1], np.ravel(n3_1*v_1)))
    a2d1.plot(xdata_1, zdata_1)
    a2d1.set_xlabel('X (x1)')
    a2d1.set_ylabel('Z (x3)')

#    {x2, x3}   orth to x1 ((Az = 0, fi = 90))
    ind_2 = (azm == np.radians(90))
    v_2 = vlist[ind_2]
    n2_2 = n2[ind_2]
    n3_2 = n3[ind_2]
    xdata_2 = np.hstack((-np.ravel(n2_2*v_2)[::-1], np.ravel(n2_2*v_2)))
    zdata_2 = -1 * np.hstack((np.ravel(n3_2*v_2)[::-1], np.ravel(n3_2*v_2)))
    a2d2.plot(xdata_2, zdata_2)
    a2d2.set_xlabel('Y (x2)')
    a2d2.set_ylabel('Z (x3)')
    
#    {x1, x2}   orth to x3 (Map view)
    ind_3 = (anm == np.max(th))
    v_3 = vlist[ind_3]
    n1_3 = n1[ind_3]
    n2_3 = n2[ind_3]
    xdata_3 = np.hstack((-np.ravel(n1_3*v_3)[::-1], np.ravel(n1_3*v_3)))
    zdata_3 = -1 * np.hstack((np.ravel(n2_3*v_3)[::-1], np.ravel(n2_3*v_3)))
    a2d3.plot(xdata_3, zdata_3)
    a2d3.set_xlabel('X (x1)')
    a2d3.set_ylabel('Y (x2)')  
    
    f2d.tight_layout()


    a2d1.set_xlim([-maxlim,maxlim])        
    a2d1.set_ylim([-maxlim,0])        
    a2d2.set_xlim([-maxlim,maxlim])        
    a2d2.set_ylim([-maxlim,0])        
    a2d3.set_xlim([-maxlim,maxlim])        
    a2d3.set_ylim([-maxlim,maxlim])     
    
    for ax in [a2d1, a2d2, a2d3]:
        ax.grid(True, c=[0.4,0.4,0.4])


an=np.linspace(0,90,16)[:-1]
#az=np.hstack((np.linspace(0,180-180/20,20),0))
az=np.linspace(0,360,25)[:-1]

th=an*pi/180.0
fi=pi/2 - az*pi/180



alpha1=2000.0
beta1=1000.0
rho1=2000.0
alpha2=4000.0
beta2=1400.0
rho2=2075.0


de1=-0.2
ep1=0.05
ga1=0.22
de2 = 0.3
ep2 = 0.15
ga2 = 0.04
de3 = -0.05



C1 = cij_vti(alpha1, beta1, rho1, de1, ep1, ga1)
C1_vti_from_ort = cij_ort(alpha1, beta1, rho1, de1, ep1, ga1, de1, ep1, ga1, 0)
C1_hti_from_ort = cij_ort(alpha1, beta1, rho1, 0, 0, 0, de1, ep1, ga1, de1)
C1_ort = cij_ort(alpha1, beta1, rho1, de1, ep1, ga1, de2, ep2, ga2, de3)
C1=C1_ort


vlist=np.ones((len(az), len(an)))
vslist=np.ones((len(az), len(an)))
vtlist=np.ones((len(az), len(an)))

for i, az_inc in enumerate(fi):
    for j, ang_inc in enumerate(th):
        L1 = L_matrix(C1, rho1, az_inc, ang_inc)
        V = count_v_Christoffel(L1)
        vlist[i,j] = V[0]
        vslist[i,j] = V[1]
        vtlist[i,j] = V[2]

#maxlim = np.max(vlist)

#fig1 = plt.figure(facecolor='w')
#ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
#ax1.set_theta_zero_location("S")
#ax1.plot(pi*an/180, vlist[0,:], lw=2)

#fig2 = plt.figure(facecolor='w')
#ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
#ax2.plot(an,vlist[0,:], lw=2)
#ax2.set_xlabel('Incident Angle')

#fig4 = plt.figure(facecolor='w')
#ax4 = fig4.add_axes([0.1,0.1,0.8,0.8])
#ax4.plot(az,vlist[:,0], lw=2)
#ax4.set_xlabel('Azimuth (geo)')






#==---PLOTTING 3D SURFACE
#anm,azm=np.meshgrid(th, fi)
#
#
#n1 = cos(azm)*sin(anm)
#n2 = sin(azm)*sin(anm)
#n3 = cos(anm)
##
#
#fig3 = plt.figure(facecolor='w')
#ax3 = fig3.add_axes([0.1,0.1,0.8,0.8],projection='3d')
#
##ax3.plot_surface(vlist*n1, vlist*n2, -vlist*n3, alpha=0.4, 
##                 rstride=1, cstride=1, 
##                 facecolors = cm.Spectral_r((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))), 
###                 edgecolor = ((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))), 
##                 edgecolors = 'k', 
##                 linewidths=0.5, antialiased=False, shade=False)
#
##ax3.plot_wireframe(vlist*n1, vlist*n2, -vlist*n3, alpha=0.4, 
##                   rstride=1, cstride=1, 
##                   color = cm.Spectral_r((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))), 
###                   edgecolor = ((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))), 
###                   color = 'k', 
##                   linewidths=1, antialiased=False)
#
#ax3.scatter(vlist*n1, vlist*n2, vlist*n3, c = ((vlist-np.min(vlist))/(np.max(vlist)-np.min(vlist))),
#            cmap='Spectral_r', alpha=0.8, marker='o', s = 10, edgecolors = 'none', linewidths=0)
#
#
#ax3.set_xlabel('X')
#ax3.set_ylabel('Y')
#ax3.set_zlabel('Z')
#
#
##cset = ax3.plot(xs = vslist[:,20]*cos(pi/2-fi), ys = vslist[:,20]*sin(pi/2-fi), zs = -np.max(vslist), zdir='z')
#
#
#xc,yc,zc = get_cube_points(maxlim)
#ax3.scatter(xc, yc, zc, alpha=0)
#ax3.scatter(0,0,0, alpha=0.7, s=1)
#ax3.set_xlim([-maxlim, maxlim])
#ax3.set_ylim([-maxlim, maxlim])
#ax3.set_zlim([-maxlim, maxlim])
#
##axisEqual3D(ax3)
#ax3.set_aspect('equal')





fancy_aniso_plot(th, fi, vlist)