from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from diffeq_dodane import *
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
import scipy.special as spec
from numpy import linalg as la
from timeit import default_timer as timer

def forcel(y):
    return -sin(y) #-omega2*y

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def energy_lin(y,v):
    return (1*y*y+v*v)/2.
def energy(theta,dth,omega):
    return 1-cos(theta)+ (dth**2)/(2*omega**2)

#-----------------------------------------------------------------------------
# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state,t):

    dydt=np.zeros_like(state)
    dydt[0]=state[1] # x' = v
    dydt[1]=forcel(state[0])  # v' = F(x)

    return dydt


############analiticna resitev##########
def resitev(t, th_0, omega):
    res = 2*np.arcsin(np.sin(th_0/2)*spec.ellipj(spec.ellipk(np.sin(th_0/2)**2)-omega*t,np.sin(th_0/2)**2)[0])
    return res


def casovna_odv():
    dt =  0.002
    t = np.arange(0.0, 100, dt)
    th0=1.
    dth=0.
    w=1
    iconds=np.array([th0,dth])
    x_scipy=integrate.odeint(pendulum,iconds,t)
    x_euler=euler(pendulum,iconds,t)
    x_rk4=rku4(pendulum,iconds,t)
    x_verlet=verlet(forcel,th0,dth,t)
    x_pefrl=pefrl(forcel,th0,dth,t)
    x_pc4=pc4(pendulum, [th0,dth],t)
    x_rk45 = rk45(pendulum,[th0,dth],t)
    fig, ax = plt.subplots(2,2,sharex=True)
    ax[0][0].plot(t,x_scipy[:,0], label='scipy')
    ax[0][0].plot(t, resitev(t, th0, w), label='analiticna')
    ax[0][0].plot(t, x_euler[:,0], label='euler')
    ax[0][0].plot(t, x_rk4[:,0], label='rk4')
    ax[0][0].plot(t, x_verlet[0,:], label='verlet')
    ax[0][0].plot(t, x_pefrl[0,:], label='pefrl')
    ax[0][0].plot(t, x_rk45[0][:,0], label='rk45')
    ax[0][0].plot(t, x_pc4[:,0], label='pc4')
    ax[0][0].set_title(r'$resitve \ \ddot{\theta} = -sin(\theta)$')
    ax[0][0].legend()
    en_scipy=energy(x_scipy[:,0],x_scipy[:,1], w)
    en_euler=energy(x_euler[:,0],x_euler[:,1], w)
    en_rk4=energy(x_rk4[:,0],x_rk4[:,1], w)
    en_pc4 = energy(x_pc4[:,0], x_pc4[:,1], w)
    en_rk45=energy(x_rk45[0][:,0],x_rk45[0][:,1], w)
    en_pefrl=energy(x_pefrl[0,:],x_pefrl[1,:], w)
    en_verlet=energy(x_verlet[0,:],x_verlet[1,:], w)
    ax[0][1].plot(t, en_scipy-en_pefrl, label='scipy-pefrl')
    ax[0][1].plot(t, en_euler-en_pefrl, label='euler-pefrl')
    ax[0][1].plot(t, en_rk4-en_pefrl, label='rk4-pefrl')
    ax[0][1].plot(t, en_verlet-en_pefrl, label='verlet-pefrl')
    ax[0][1].set_title(r'$ \Delta E$')
    ax[0][1].legend()
    ax[1][1].plot(t, en_scipy-en_pefrl, label='scipy-pefrl')
    ax[1][1].plot(t, en_rk4-en_pefrl, label='rk4-pefrl')
    ax[1][1].plot(t, en_verlet-en_pefrl, label='verlet-pefrl')
    ax[1][1].plot(t, en_rk45-en_pefrl, label='rk45-pefrl')
    ax[1][1].plot(t, en_pc4-en_pefrl, label='pc4-pefrl')

    ax[1][1].legend()
    #ax[1][0].plot(t, x_rk45[0][:,0]-resitev(t, th0, w), label='rk45')
    ax[1][0].plot(t,x_scipy[:,0]-resitev(t, th0, w), label='scipy')
    #ax[1][0].plot(t, x_rk4[:,0]-resitev(t, th0, w), label='rk4')
    #ax[1][0].plot(t, x_pc4[:,0]-resitev(t, th0, w), label='pc4')
    #ax[1][0].plot(t, x_verlet[0,:]-resitev(t, th0, w), label='verlet')
    #ax[1][0].plot(t, x_pefrl[0,:]-resitev(t, th0, w), label='pefrl')
    ax[1][0].set_title(r'$\Delta \theta$')
    ax[1][0].legend()
    fig.supxlabel('t')
    fig.tight_layout()
    fig.suptitle('dt=0.002')
    plt.plot()
    plt.show()
    return None
casovna_odv()

def casi():
    dt = np.logspace(0.0, -4.0, num=20)
    #t = np.arange(0.0, 150, dt)
    th0=1.
    dth=0.
    w=1
    iconds=np.array([th0,dth])
    napaka_scipy = np.array([])
    napaka_euler = np.array([])
    napaka_rk4 = np.array([])
    napaka_verlet = np.array([])
    napaka_pefrl = np.array([])
    napaka_rk45 = np.array([])
    napaka_pc4 = np.array([])
    for i in dt:
        t = np.linspace(0,600*i, num=600)
        x_scipy=integrate.odeint(pendulum,iconds,t)
        x_euler=euler(pendulum,iconds,t)
        x_rk4=rku4(pendulum,iconds,t)
        x_verlet=verlet(forcel,th0,dth,t)
        x_pefrl=pefrl(forcel,th0,dth,t)
        x_pc4=pc4(pendulum, [th0,dth],t)
        x_rk45 = rk45(pendulum,[th0,dth],t)
        res = resitev(t, th0, w)
        napaka_scipy = np.append(napaka_scipy, la.norm(x_scipy[:,0]-res))
        napaka_verlet = np.append(napaka_verlet, la.norm(x_verlet[0,:]-res))
        napaka_euler = np.append(napaka_euler, la.norm(x_euler[:,0]-res))
        napaka_rk4 = np.append(napaka_rk4, la.norm(x_rk4[:,0] - res))
        napaka_pefrl = np.append(napaka_pefrl, la.norm(x_pefrl[0,:]-res))
        napaka_pc4 = np.append(napaka_pc4, la.norm(x_pc4[:,0]-res))
        napaka_rk45 = np.append(napaka_rk45, la.norm(x_rk45[0][:,0]-res))
    plt.plot(dt, napaka_rk4, label='rk4')
    plt.plot(dt, napaka_scipy, label='scipy')
    plt.plot(dt, napaka_euler, label='euler')
    plt.plot(dt, napaka_pefrl, label='pefrl')
    plt.plot(dt, napaka_rk45, label='rk45')
    plt.plot(dt, napaka_pc4, label='pc4')
    plt.plot(dt, napaka_verlet, label='verlet')
    plt.legend()
    plt.xlabel('korak')
    plt.ylabel(r'$||napaka||_{2}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    return None
def casovna_zahtevnost():
    dt = np.logspace(0.0, -4.0, num=20)
    #t = np.arange(0.0, 150, dt)
    th0=1.
    dth=0.
    w=1
    iconds=np.array([th0,dth])
    cas_scipy = np.array([])
    cas_euler = np.array([])
    cas_rk4 = np.array([])
    cas_ver = np.array([])
    cas_pefrl = np.array([])
    cas_rk45 = np.array([])
    cas_pc4 = np.array([])
    fig, ax = plt.subplots(1)
    for i in dt:
        t = np.arange(0,10, i)
        tz_sc = timer()
        x_scipy=integrate.odeint(pendulum,iconds,t)
        tk_sc=timer()
        cas_scipy=np.append(cas_scipy, tk_sc-tz_sc)
        tz_eu = timer()
        x_euler=euler(pendulum,iconds,t)
        tk_eu=timer()
        cas_euler=np.append(cas_euler, tk_eu-tz_eu)
        tz_rk4 = timer()
        x_rk4=rku4(pendulum,iconds,t)
        tk_rk4=timer()
        cas_rk4=np.append(cas_rk4, tk_rk4-tz_rk4)
        tz_ver = timer()
        x_verlet=verlet(forcel,th0,dth,t)
        tk_ver=timer()
        cas_ver=np.append(cas_ver, tk_ver-tz_ver)
        tz_pef = timer()
        x_pefrl=pefrl(forcel,th0,dth,t)
        tk_pef=timer()
        cas_pefrl=np.append(cas_pefrl, tk_pef-tz_pef)
        tz_pc4 = timer()
        x_pc4=pc4(pendulum, [th0,dth],t)
        tk_pc4=timer()
        cas_pc4=np.append(cas_pc4, tk_pc4-tz_pc4)
        tz_rk45 = timer()
        x_rk45 = rk45(pendulum,[th0,dth],t)
        tk_rk45=timer()
        cas_rk45=np.append(cas_rk45, tk_rk45-tz_rk45)
    plt.plot(dt, cas_scipy, label='scipy')
    plt.plot(dt, cas_pc4, label='pc4')
    plt.plot(dt, cas_rk4, label='rk4')
    plt.plot(dt, cas_rk45, label='rk45')
    plt.plot(dt, cas_ver, label='ver')
    plt.plot(dt, cas_pefrl, label='pefrl')
    plt.plot(dt, cas_euler, label='euler')
    plt.legend()
    plt.title(r'$\ddot{\theta} = -sin(\theta)$, t=[0,10] ')
    plt.xlabel('korak')
    plt.ylabel('t[s]')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def napake_kota():
    dt =  0.02
    t = np.arange(0.0, 100, dt)
    th0=1.
    dth=0.
    w=1
    iconds=np.array([th0,dth])
    x_scipy=integrate.odeint(pendulum,iconds,t)
    x_euler=euler(pendulum,iconds,t)
    x_rk4=rku4(pendulum,iconds,t)
    x_verlet=verlet(forcel,th0,dth,t)
    x_pefrl=pefrl(forcel,th0,dth,t)
    x_pc4=pc4(pendulum, [th0,dth],t)
    x_rk45 = rk45(pendulum,[th0,dth],t)
    fig, ax = plt.subplots(3)

    ax[0].plot(t, x_rk45[0][:,0]-resitev(t, th0, w), label='rk45')
    ax[0].plot(t,x_scipy[:,0]-resitev(t, th0, w), label='scipy')
    ax[0].plot(t, x_rk4[:,0]-resitev(t, th0, w), label='rk4')
    ax[0].plot(t, x_pc4[:,0]-resitev(t, th0, w), label='pc4')
    ax[0].plot(t, x_pefrl[0,:]-resitev(t, th0, w), label='pefrl')
    ax[0].text(40,1e-5, 'dt=0.02')
    ax[0].legend()
    ax[0].set_ylabel(r'$\Delta \theta$')
    ax[0].set_yscale('log')
    dt =  1
    t = np.arange(0.0, 1000, dt)
    th0=1.
    dth=0.
    w=1
    iconds=np.array([th0,dth])
    x_scipy=integrate.odeint(pendulum,iconds,t)
    x_euler=euler(pendulum,iconds,t)
    x_rk4=rku4(pendulum,iconds,t)
    x_verlet=verlet(forcel,th0,dth,t)
    x_pefrl=pefrl(forcel,th0,dth,t)
    x_pc4=pc4(pendulum, [th0,dth],t)
    x_rk45 = rk45(pendulum,[th0,dth],t)

    en_scipy=energy(x_scipy[:,0],x_scipy[:,1], w)
    en_rk4=energy(x_rk4[:,0],x_rk4[:,1], w)
    en_pc4 = energy(x_pc4[:,0], x_pc4[:,1], w)
    en_rk45=energy(x_rk45[0][:,0],x_rk45[0][:,1], w)
    en_pefrl=energy(x_pefrl[0,:],x_pefrl[1,:], w)
    en_verlet=energy(x_verlet[0,:],x_verlet[1,:], w)
    ax[1].plot(t, en_scipy, 'k',label='scipy')
    ax[1].text(400,2,'dt=1')
    #ax[1].plot(t, en_pc4, label='pc4')
    ax[1].plot(t, en_rk4, label='rk4')
    ax[1].plot(t, en_rk45, label='rk45')
    #ax[1].plot(t, en_verlet, label='verlet')
    ax[1].plot(t, en_pefrl, label='pefrl')
    ax[1].set_ylabel(r'$E$')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[2].plot(t, en_scipy, 'k',label='scipy')
    #ax[2].plot(t, en_verlet, label='verlet')
    #ax[2].plot(t, en_pefrl, label='pefrl')
    ax[2].set_ylabel(r'$E$')
    ax[2].set_yscale('linear')
    ax[2].legend()

    fig.supxlabel('t')
    fig.tight_layout()
    plt.show()
napake_kota()
