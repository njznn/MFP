from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from diffeq_dodane import *
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

########## c++ rk4#######
t, x, y = np.loadtxt('/home/ziga/Desktop/C++/numericne_metode/data.txt', skiprows=1, usecols=(0, 1, 2), unpack=True)
#plt.plot(t,y)


omega2=1

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(y):
    return  -sin(y) #omega2*y

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def energy(y,v):
    return (omega2*y*y+v*v)/2.

#-----------------------------------------------------------------------------
# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state,t):

    dydt=np.zeros_like(state)
    dydt[0]=state[1] # x' = v
    dydt[1]=forcel(state[0])  # v' = F(x)

    return dydt
def duseno_mat(y,t, beta=0.5):
    return [y[1],-sin(y[0]) - beta*y[1]]

def duseno_mat_vzb(y,t,beta=0.5,v=1.5, w=2/3):
    return [y[1],v*cos(w*t)-sin(y[0]) - beta*y[1]]

def van_der_pool(y,t,):
    return [y[1], 3*y[1]*(1-y[0]**2) - y[0] + 1*cos(t)]

def fazni_diagram_colormesh():
    dt =  0.2
    t = np.arange(0.0, 100, dt)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    y1 = np.linspace(-3.0, 8.0, 30)
    y2 = np.linspace(-3.0, 3.0, 30)

    Y1, Y2 = np.meshgrid(y1, y2)

    t = 0

    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

    NI, NJ = Y1.shape

    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            yprime = pendulum([x, y], t)
            u[i,j] = yprime[0]
            v[i,j] = yprime[1]


    Q = plt.quiver(Y1, Y2, u, v, color='r')

    plt.xlabel(r'$\theta(t)$')
    plt.ylabel(r"$\theta' (t)$")
    plt.xlim([-3, 8])
    plt.ylim([-3, 3])
    theta = np.arange(0, 2.7, 0.01)
    normalize = mcolors.Normalize(vmin=0, vmax=2.7)
    colormap = cm.jet
    for y20 in theta:
        tspan = np.linspace(0, 50, 200)
        y0 = [1.0, y20]
        ys = integrate.odeint(pendulum, y0, tspan)
        ax1.plot(ys[:,0], ys[:,1],color=colormap(normalize(y20))  )
        ax1.plot(-ys[:,0], ys[:,1], color=colormap(normalize(y20))) # path
        plt.plot(-ys[:,0], -ys[:,1], color=colormap(normalize(y20)))
        plt.plot(ys[:,0], -ys[:,1], color=colormap(normalize(y20)))
        plt.plot(ys[:,0]+2*np.pi, ys[:,1], color=colormap(normalize(y20))) # path
        #plt.plot([ys[0,0]], [ys[0,1]], color=colormap(normalize(y20))) # start
        #plt.plot([ys[-1,0]], [ys[-1,1]], 's') # end

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(10)
    plt.colorbar(scalarmappaple, label=r"$\theta' (0)$")
    plt.legend()
    plt.show()
    return None
fazni_diagram_colormesh()
def fazni_diagram():
    dt =  0.2
    t = np.linspace(0, 10, 200)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    y1 = np.linspace(-10.0, 10.0, 30)
    y2 = np.linspace(-10.0, 10.0, 30)

    Y1, Y2 = np.meshgrid(y1, y2)

    t = 0

    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

    NI, NJ = Y1.shape

    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            yprime = duseno_mat_vzb([x, y], t)
            u[i,j] = yprime[0]
            v[i,j] = yprime[1]


    Q = plt.quiver(Y1, Y2, u, v, color='r')

    plt.xlabel(r'$x(t)$')
    plt.ylabel(r"$x' (t)$")
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    theta = np.arange(0, 3, 0.5)
    for y20 in theta:
        tspan = np.linspace(0, 20, 200)
        y0 = [1.0, y20]
        ys = integrate.odeint(pendulum, y0, tspan)
        plt.plot(ys[:,0], ys[:,1],'-' )
        #ax1.plot(-ys[:,0], ys[:,1], 'k') # path
        #plt.plot(-ys[:,0], -ys[:,1], 'b')
        #plt.plot(ys[:,0], -ys[:,1], 'b')
        #plt.plot(ys[:,0]+2*np.pi, ys[:,1], 'b') # path
        #plt.plot([ys[0,0]], [ys[0,1]], 'o') # start
        #plt.plot([ys[0,0]], [-ys[0,1]], 'o') # start
        #plt.plot([ys[-1,0]], [ys[-1,1]], 's') # end

    plt.title(r'$\ddot{x} + 0.5\dot{x} + sin(x) =  2cos(t)$')
    #plt.title(r'$\ddot{x} - 3\dot{x}(1-x^2) + x =  cos(t)$')
    plt.legend()
    plt.show()
    return None
fazni_diagram()
