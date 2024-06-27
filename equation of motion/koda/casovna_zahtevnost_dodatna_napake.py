import numpy as np
import matplotlib.pyplot as plt
from diffeq_metode import *
import matplotlib.patches as mpatches
from timeit import default_timer as timer
import pylab


def casovna_zahtevnost():
    euler1 = np.array([])
    heun1 = np.array([])
    rk2a1 = np.array([])
    rk2b1 = np.array([])
    rku41 = np.array([])
    rk451 = np.array([])
    rkf1 = np.array([])
    pc41 = np.array([])
    for i in range(10, 10000, 10):
        k = 0.06
        T_zun = 10.0
        def f(T, t):
            return (-k*(T-T_zun))
        a, b = ( 0.0, 100)
        T0 = 25.0
        n =100
        korak = 100/i
        t = np.linspace( a, b, n )
        se = timer()
        euler(f,T0, t)
        ee = timer()
        euler1 =np.append(euler1, ee-se)
        sh = timer()
        heun(f,T0, t)
        eh = timer()
        heun1 =np.append(heun1, eh-sh)
        s2a = timer()
        rk2a(f,T0, t)
        e2a = timer()
        rk2a1 =np.append(rk2a1, e2a-s2a)
        s2b = timer()
        rk2b(f,T0, t)
        e2b = timer()
        rk2b1 =np.append(rk2b1, e2b-s2b)
        srk4 = timer()
        rku4(f,T0, t)
        erk4 = timer()
        rku41 =np.append(rku41, erk4-srk4)
        srk45 = timer()
        rk45(f,T0, t)
        erk45 = timer()
        rk451 =np.append(rk451, erk45-srk45)
        srkf = timer()
        rkf(f,a,b,T0, 1e-5, 1.0,0.01)
        erkf = timer()
        rkf1 =np.append(rkf1, erkf-srkf)
        spc = timer()
        pc4(f,T0, t)
        epc = timer()
        pc41 =np.append(pc41, epc-spc)
    x = [i for i in range(10,10000,10)]
    print('euler:', '{:.2e}'.format(np.average(euler1)))
    print('heun:', '{:.2e}'.format(np.average(heun1)))
    print('rk2a:', '{:.2e}'.format(np.average(rk2a1)))
    print('rk2b:', '{:.2e}'.format(np.average(rk2b1)))
    print('rku4:', '{:.2e}'.format(np.average(rku41)))
    print('rk45:', '{:.2e}'.format(np.average(rk451)))
    print('rkf:', '{:.2e}'.format(np.average(rkf1)))
    print('pc4:', '{:.2e}'.format(np.average(pc41)))
    return None
casovna_zahtevnost()
def napaka_od_velikosti():
    n_vred= np.logspace(1,5,10)
    n_vred = n_vred[::-1]
    najvecja = np.array([])
    najvecja_euler = np.array([])
    heun1 = np.array([])
    rk2a1 = np.array([])
    rk2b1 = np.array([])
    rku41 = np.array([])

    pc41 = np.array([])
    print(n_vred)
    for i in range(10):
        k = 0.06
        T_zun = 10.0
        def f(T, t):
            return (-k*(T-T_zun))
        max = 100
        a, b = ( 0.0, max)
        T0 = 25.0
        n =int(n_vred[i])
        h = max/n
        t = np.linspace( a, b, n )
        prava = T_zun + np.exp(-k*t)*(T0-T_zun)
        napaka = np.max(abs(rk45(f,T0, t)[0] - prava))
        napaka_heun = np.max(abs(heun(f,T0, t) - prava))
        napakae = np.max(abs(euler(f,T0,t) - prava))
        napakark2a = np.max(abs(rk2a(f,T0,t) - prava))
        napakark2b = np.max(abs(rk2b(f,T0,t) - prava))
        napakarku4 = np.max(abs(rku4(f,T0,t) - prava))
        napakapc4 = np.max(abs(pc4(f,T0,t) - prava))
        pc41 = np.append(pc41, napakapc4)
        rku41 = np.append(rku41, napakarku4)
        najvecja = np.append(najvecja, napaka)
        heun1 = np.append(heun1, napaka_heun)
        rk2a1 = np.append(rk2a1, napakark2a)
        rk2b1 = np.append(rk2b1, napakark2b)
        najvecja_euler = np.append(najvecja_euler, napakae)
    fig, ax = plt.subplots()
    plt.yscale('log')
    plt.xlabel('velikost koraka')
    plt.ylabel('maksimalna absolutna napaka')

    plt.title(r" $k = 0.06,T_{0} = 25,T_{zun} = 10$")
    plt.xscale('log')
    plt.plot(max/n_vred, najvecja_euler, label='euler')
    plt.plot(max/n_vred, heun1, label='heun')
    plt.plot(max/n_vred, rk2a1, label='rk2a')
    plt.plot(max/n_vred, rk2b1, label='rk2b')
    plt.plot(max/n_vred, rku41, label='rku4')
    plt.plot(max/n_vred, najvecja, label='rk45')
    plt.plot(max/n_vred, pc41, label='pc41')
    plt.legend()
    plt.show()
    return None

################# Dodatna naloga###############
'''
fig, ax = plt.subplots()
for j in range(10, 11, 1):
    j = j/10
    for i in range(1, 11, 1):
        i = i/10
        delta = 10.0
        A = j
        T_zun = 0.0
        k =  i
        def f(T, t):
            return(-k*(T-T_zun)) + A*np.sin((2*np.pi/24)*(t-10))
        a, b = ( 0.0, 100)
        T0 = 15.0
        n =100
        h = 1
        t = np.linspace( a, b, n )
        plt.plot(t, rk45(f,T0, t)[0], label='k = {}'.format(i))
    pylab.text(0.5, 1.0, " A = {}".format(j), transform=ax.transAxes)
    plt.xlabel('t')
    plt.ylabel('T')
    plt.legend(loc='upper center', bbox_to_anchor=(1.0, 1.0))
    fig.savefig("/home/ziga/Desktop/FMF/3.letnik/MFP/dodatna.png".format(j))
    plt.clf()
'''
