import numpy as np
from diffeq_metode import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab

class resitev:
    def __init__(self, f, x0, t):
        self.x0 = x0
        self.t = t
        self.f=f
    def narisi_euler(self):
        plt.plot(self.t, euler(self.f,self.x0,self.t), '-o', markersize = '3')
        return None
    def narisi_heun(self):
        plt.plot(self.t, heun(self.f,self.x0,self.t),'--', label='heun')
        return None
    def narisi_rk2a(self):
        plt.plot(self.t, rk2a(self.f,self.x0,self.t), label='rk2a')
        return None
    def narisi_rk2b(self):
        plt.plot(self.t, rk2a(self.f,self.x0,self.t), label='rk2b')
        return None
    def narisi_rku4(self):
        plt.plot(self.t, rku4(self.f, self.x0, self.t), label='rku4')
        return None
    def narisi_rk45(self):
        plt.plot(self.t, rk45(self.f, self.x0, self.t)[0], label='rku45')
        return(None)
    def pokazi_in_pocisti(self):
        plt.show()
        plt.clf()
        return None



###############interval, st korakov, zacetni pogoj#####
fig, ax = plt.subplots()
for i in range(99,100, 1):
    k = 0.06
    T_zun = -5.0
    def f(T, t):
        return (-k*(T-T_zun))
    a, b = ( 0.0, 100)
    T0 = -15.0
    n =i
    korak = 100/i
    t = np.linspace( a, b, n )

    resitev(f,T0,t).narisi_euler()
    resitev(f,T0,t).narisi_rk2a()
    resitev(f,T0,t).narisi_heun()
    resitev(f,T0,t).narisi_rk2b()
    resitev(f,T0,t).narisi_rku4()
    resitev(f,T0,t).narisi_rk45()
plt.xlabel('t')
plt.ylabel('T')
plt.title('rešitve')
pylab.text(0.6, 0.8, " k = 0.06 \n T_0 = -15 \n T_zun = -5", transform=ax.transAxes)
plt.grid(True)
#plt.plot(t, T_zun + np.exp(-k*t)*(T0-T_zun), label="T(t)")
plt.legend()
#resitev(f,T0,t).pokazi_in_pocisti()
plt.clf()
########################## ANIMACIJA#########
'''
for i in range(8,10, 2):
    i = i/100
    k = i
    T_zun = -5.0
    def f(T, t):
        return (-k*(T-T_zun))
    a, b = ( 0.0, 100)
    T0 = -15.0
    n =10000
    h = 1
    t = np.linspace( a, b, n )
    prava = T_zun + np.exp(-k*t)*(T0-T_zun)
    pravaf = lambda t: T_zun + np.exp(-k*t)*(T0-T_zun)
    fig, ax = plt.subplots(5, 2, sharex=True)
    plt.rcParams.update({'font.size': 8})
    ax[0][0].plot(t, euler(f, T0, t), 'o-', markersize='3',label='Euler')
    ax[2][0].set_ylabel('T')
    ax[0][0].text(0.75, 0.4, "k = {} \n h=1".format(np.round(i,2)), transform=ax[0][0].transAxes)
    ax[0][0].set_title("eksplicitna metoda 1.reda")
    ax[1][0].set_title("eksplicitne metode 2.reda")
    ax[2][0].set_title("eksplicitne metode 4.reda")
    ax[3][0].set_title("adaptivna metoda RK-Fehlberg")
    ax[4][0].set_title("P-C Adams-Bashforth-Moulton 4.reda")
    ax[0][1].set_title("absolutna napaka")
    ax[0][1].plot(t, abs(euler(f,T0,t) - prava), 'o-', markersize='3')
    ax[2][1].set_ylabel(r'|$T_{p}-T$|')
    ax[0][0].legend(frameon=False)
    ax[1][0].plot(t, heun(f,T0,t),'o-', markersize='3' ,label='Heun' )
    ax[1][0].plot(t, rk2a(f,T0, t),'o-', markersize='3' ,label='rk2a')
    ax[1][0].plot(t, rk2b(f,T0, t),'o-', markersize='3' ,label='rk2b')
    ax[1][0].text(0.4, 0.6, "k = {} \n h=1".format(np.round(i, 2)), transform=ax[1][0].transAxes)
    ax[1][1].plot(t, abs(heun(f,T0,t) - prava), 'o-', markersize='3', label='Heun')
    ax[1][1].plot(t,abs(rk2a(f,T0, t) - prava), 'o-', markersize='3', label='rk2a')
    ax[1][1].plot(t, abs(rk2b(f,T0, t) - prava), 'o-', markersize='3', label='rk2b')
    ax[1][0].legend(frameon=False)
    ax[1][1].legend(frameon=False)
    ax[2][0].plot(t, rku4(f,T0,t),'o-', markersize='3' ,label='rku4' )
    ax[2][0].plot(t, rk45(f,T0,t)[0],'o-', markersize='3' ,label='rku45' )
    ax[2][0].text(0.5, 0.2, "k = {} \n h=1".format(np.round(i,2)), transform=ax[2][0].transAxes)
    ax[2][1].plot(t, abs(rku4(f,T0,t)-prava),'o-', markersize='3' ,label='rku4' )
    ax[2][1].plot(t, abs(rk45(f,T0,t)[0]-prava),'o-', markersize='3' ,label='rku45' )
    ax[2][0].legend(frameon=False)
    ax[2][1].legend(frameon=False)
    ax[3][0].plot(rkf(f,a,b, T0, 1e-4, h, 0.01)[0],rkf(f,a,b, T0, 1e-4, h, 0.01)[1], 'mo-', markersize='3' ,label='rkf' )
    ax[3][0].text(0.5, 0.5, "k = {}, h_min=0.01 \n h_max = 1 , tol=1e-4".format(np.round(i,2)), transform=ax[3][0].transAxes)
    ax[3][1].plot(rkf(f,a,b, T0, 1e-4, h, 0.01)[0],abs(rkf(f,a,b, T0, 1e-4, h, 0.01)[1] - pravaf(rkf(f,a,b, T0, 1e-4, h, 0.01)[0])), 'mo-', markersize='3' )
    ax[4][0].plot(t,pc4(f,T0,t), 'ro-', markersize='3' ,label='rkf' )
    ax[4][0].text(0.5, 0.5, "k = {} \n h=1".format(np.round(i, 2)), transform=ax[4][0].transAxes)
    ax[4][1].plot(t,abs(pc4(f,T0,t) - prava), 'ro-', markersize='3' )

    ax[4][0].set_xlabel('t')
    ax[4][1].set_xlabel('t')
    fig.tight_layout()
    fig.set_size_inches(6.5, 7.5)
    #fig.savefig("/home/ziga/Desktop/FMF/3.letnik/MFP/resitve_napake__obrat{}.png".format(i))
    plt.show()
'''
