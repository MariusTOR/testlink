# ENAE788I Final Project
# Problem 11
#
# Br. Marius Strom, TOR
# Modal analysis based on a code from Dr. James Baeder

# This code computes the 1st and 2nd fore-aft tower mode frequencies and then the tower top deflections at the turbine's operating point.  It is assumed that the fore-aft, side-side, and torsional modes are uncoupled, that deflections are small, and that there are no unsteady forces transmitted to the tower

# The tower's fore-aft natural frequencies will be computed first, and then its deflections

import numpy as np
import scipy.interpolate as spi
import scipy.linalg as spl
import matplotlib.pyplot as plt
import time

##########################################
# Define shape functions and derivatives #
##########################################

def shapefun(x,i,length):
    "This contains the 4 shape functions for each element"
    if i==0:
        H=2*x**3-3*x**2+1
    elif i==1:
        H=(x**3-2*x**2+x)*length
    elif i==2:
        H=-2*x**3+3*x**2
    elif i==3:
        H=(x**3-x**2)*length
    return H

def shapefundH(x,i,length):
    "This contains the 4 shape functions*d/dx for each element"
    l1=1/length
    if i==0:
        dH=(6*x**2-6*x)*l1
    elif i==1:
        dH=3*x**2-4*x+1
    elif i==2:
        dH=(-6*x**2+6*x)*l1
    elif i==3:
        dH=3*x**2-2*x**1
    return dH

def shapefunddH(x,i,length):
    "This contains the 4 shape functions*d2/dx2 for each element"
    l1=1/length
    l2=l1*l1
    if i==0:
        ddH=(12*x-6)*l2
    elif i==1:
        ddH=(6*x-4)*l1
    elif i==2:
        ddH=(-12*x+6)*l2
    elif i==3:
        ddH=(6*x-2)*l1
    return ddH

##############################################
# Definition of material constants and tower #
##############################################

# Materials constants and geometry (from NREL report)
# Based on a notional land-based design that would be altered to fit the offshore installation strategy.  Linear variation bottom->top?

filename='towerprops.txt'
towerdata=np.loadtxt(filename,skiprows=3)
height=towerdata[:,0]  # [m] tower height stations
dens=towerdata[:,2]    # [kg/m] sectional mass density
FAstiff=towerdata[:,3]*1E9 # [Nm^2] sectional fore/aft bending stiffness [EI]
SSstiff=towerdata[:,4]*1E9 # [Nm^2] sectional side/side bending stiffness [EI]
GJstiff=towerdata[:,5]*1E9 # [Nm^2] sectional torsional stiffness (GJ)

g=9.807 # [m/s^2] gravitational acceleration constant

dbase=6 # [m] diameter at base
tbase=0.027 # [m] thickness at base
dtop=3.87 # [m] diameter at top
ttop=0.019 # [m] thickness at top
mtower=347460 # [kg] total tower mass
mcg=38.234 # [m] tower cg height

#####################################
# Mass data, moments of other items #
#####################################

# upwind & psi=90deg are positive, RHR for moments

mhub=56780 # [kg] hub mass
hhub=90 # [m] hub height
xhub=5 # [m] hub cg location (upwind of tower axis)
ashaft=5*np.pi/180 # [rad] shaft inclination to horizontal

mnac=240000 # [kg] nacelle mass
xnac=-1.9 # [m] nacelle cg location (downwind of tower axis)

mblades=3*17740 # [kg] blade mass*3 blades
xblades=20.475 # [m] blade cg along blade axis
precone=2.5*np.pi/180 # [rad] blade precone angle - upwind
opptconing=0*np.pi/180 # [rad] blade coning at op. pt.

topload=(mhub+mblades+mnac)*9.807 # [N]
# upwind & psi=90deg are positive
# assume small deflections per Euler-Bernoulli beam theory

# thrust load at tower top
toppt=7.38E5 # [N] rotor thrust at op. pt.

# moments at tower top (p - fore/aft moment)
# tower top loads are modeled as a concentrated, constant,
# force and moment applied at the top of the tower
phub=mhub*g*xhub # [N-m]
pnac=mnac*g*xnac # [N-m]
pblades=mblades*g*(xhub+np.sin(opptconing)*xblades) # [N-m]
thorz=toppt*np.cos(ashaft) # [N]
pthorz=thorz*hhub # [N-m]
tvert=toppt*np.sin(ashaft)
ptvert=tvert*hhub # [N-m]
ptotal=phub+pnac+pblades+pthorz-ptvert # [N-m]
topload=topload-tvert

###################################
# Set up for FEM to compute modes #
###################################

# Set up DOF, Gaussian points, weights for element-wise computations
nnode=len(height)-1
ndof = 4 # 2 DOF at each node, 4 nodes/element

# Standard Gaussian point distribution across an element
xgauss = ([0.03376524300000,
            0.16939530700000,
            0.38069040700000,
            0.61930959300000,
            0.83060469300000,
            0.96623475700000])
ngauss=len(xgauss)

# Gauss-Legendre weights from Dr. Baeder's FEM code
weights = ([0.08566224600000,
            0.18038078650000,
            0.23395696750000,
            0.23395696750000,
            0.18038078650000,
            0.08566224600000])

nvar=nnode+1
M = np.zeros([2*nvar,2*nvar]) # global mass matrix
K = np.zeros([2*nvar,2*nvar]) # global stiffness matrix

# Compute local mass, stiffness matrices

length=np.diff(height) # [m] length of each element

#########################################
# Develop local stiffness/mass matrices #
#########################################

for node in range(nnode):
    m = np.zeros([ndof,ndof]) # elemental mass matrix
    k = np.zeros([ndof,ndof]) # elemental stiffness matrix
    EI=spi.interp1d(np.array([height[node+1],height[node]]),np.array([FAstiff[node+1],FAstiff[node]]))
    density=spi.interp1d(np.array([height[node+1],height[node]]),np.array([dens[node+1],dens[node]]))

    for i in range(ndof):
        for j in range(ndof):
            for n in range(ngauss):

                # Compute location in element and interpolate sectional properties
                xlocal=length[node]*xgauss[n]
                x=height[node]+xlocal
                EIlocal=EI(x)

                # Compute local mass distribution
                mlocal=density(x)
                if node<nnode:
                    mupper=np.sum(density[node+1:nnode]*height[node+1:nnode])
                else:
                    mupper=topload

                # Compute shape function values
                Hi=shapefun(xgauss[n],i,length[node])
                Hj=shapefun(xgauss[n],j,length[node])
                dHi=shapefundH(xgauss[n],i,length[node])
                dHj=shapefundH(xgauss[n],j,length[node])
                ddHi=shapefunddH(xgauss[n],i,length[node])
                ddHj=shapefunddH(xgauss[n],j,length[node])

                # Compute local stiffness and mass matrices
                k[i,j]+=weights[n]*(EIlocal*ddHi*ddHj*xlocal-mupper*dHi*dHj*xlocal)
                m[i,j]+=weights[n]*(mlocal*Hi*Hj*xlocal)

    # Insert local stiffness and mass matrices into global matrices

    for i in range(ndof):
        for j in range(ndof):
            iglobal=node*2+i
            jglobal=node*2+j
            M[iglobal,jglobal]=M[iglobal,jglobal]+m[i,j]
            K[iglobal,jglobal]=K[iglobal,jglobal]+k[i,j]

############################################
# Solve for natural fore/aft bending freqs #
############################################

# Apply boundary conditions
M=M[2:nvar*2,2:nvar*2]
K=K[2:nvar*2,2:nvar*2]

w,v=spl.eigh(K,M) # [rad^2/s^2]
w=np.sqrt(w)/(2*np.pi) # [Hz]

print ''
print '**********************************'
print 'Tower Fore-aft Natural Frequencies'
print '**********************************'
print ''
print 'Fundamental frequency: %.6f Hz' %w[0]
print 'Second natural frequency: %.6f Hz' %w[1]
print ''
print 'The correct fundamental freq is ~0.32 Hz (Jonkman), but this is computed via FAST for a land-based turbine, so it should be softer than my estimate due to the FAST soil model'

#################################################################################
#                PART B: COMPUTE FORE-AFT DEFLECTION AT TOWER TOP               #
#################################################################################

# Initialize deflection matrix
mult=10
nnode=nnode*mult
wsection=np.zeros([nnode+1])
dwsection=np.zeros([nnode+1])
xstore=np.zeros([nnode+1])

for node in range(1,nnode):
    el=np.floor(node/mult) # Determine location in input tower data array
    lelm=length[el]/mult # Determine length of present element from input tower data
    hlocal=length[el]*(node%mult)/mult # Determine height above ground

    Mlocal=ptotal-thorz*hlocal # [N-m] Local bending moment
    EI=spi.interp1d(np.array([height[el+1],height[el]]),np.array([FAstiff[el+1],FAstiff[el]])) # [N-m^2] local stiffness

    MEI=Mlocal/EI(hlocal+height[el])

    wlocal=0.5*MEI*lelm**2 # [m]  moment-induced deflection at current node

    dwsection[node]=MEI*lelm+dwsection[node-1] # [rad] angular deflection at current node
    wsection[node]=wlocal+dwsection[node-1]*lelm+wsection[node-1] # [m] total deflection at current node
    xstore[node]=hlocal+height[el] # [m] height of current node

# Compute deflection at tower top (free end, no bending moment)
dwsection[node+1]=dwsection[node-1] # [rad]
wsection[node+1]=dwsection[node]*lelm+wsection[node] # [m]
xstore[node+1]=max(height) # [m]

plt.plot(xstore,wsection) # plot tower deflection vs height
plt.ylabel('Deflections (m)')
plt.xlabel('Height along tower (m)')

print ''
print '***********************************'
print 'Tower Fore-aft Deflection (at peak)'
print '***********************************'
print ''
print 'Maximum tower deflect: %.3f m' %wsection[node]
print ''
raw_input('press enter to exit')
