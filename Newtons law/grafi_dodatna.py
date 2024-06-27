from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from diffeq_dodane import *
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

omega2=1

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(y):
    return  -omega2*y #-sin(y)

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

def duseno_mat_vzb(y,t,beta=0.5,v=1.0, w=2/3):
    return [y[1],v*cos(w*t)-sin(y[0]) - beta*y[1]]

def van_der_pool(y,t,):
    return [y[1], 3*y[1]*(1-y[0]**2) - y[0] + 1*cos(t)]


def duseno_grafi():
    dt =  0.2
    t = np.arange(0.0, 100, dt)
    v = np.array([0.5,3.7, 4.3])
    afig, ax = plt.subplots(2, sharex=True)
    for i in v:
        y0 = [0.0, 0.0]
        res = integrate.odeint(duseno_mat_vzb, y0, t, args=(0.5,i, 2/3))
        ax[0].plot(t, res[:,0], label='v={:}'.format(i))
    ax[0].title.set_text(r'$\ddot{x} + 0.5\dot{x} + sin(x) =  vcos(t), \vec{\theta}(0)=[0,0]$')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel(r'$\theta$')
    ax[0].legend()
    res = integrate.odeint(duseno_mat_vzb, y0, t, args=(0.5,1.0, 2/3))
    ax[1].plot(t, res, label=['', 'odvod'])
    ax[1].title.set_text(r'$\ddot{x} + 0.5\dot{x} + sin(x) =  cos(t), \vec{\theta}(0)=[0,0]$')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel(r'$\theta, \dot{\theta}$')
    ax[1].legend()
    plt.tight_layout()
    plt.legend()
    plt.show()
    return None
def van_der_pool_graf():
    dt =  0.2
    t = np.arange(0.0, 100, dt)
    y0=[1,0]
    res = integrate.odeint(van_der_pool, y0, t)
    plt.plot(t, res[:,0], 'r')
    plt.plot(t, res[:,1], 'gold', label='odvod')
    plt.title(r'$\ddot{x} - 3\dot{x}(1-x^2) + x =  cos(t), \vec{x(0)} = [1,0]$')
    plt.xlabel('t')
    plt.ylabel(r'$x$')
    plt.legend()
    plt.show()
van_der_pool_graf()
