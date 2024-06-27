import numpy as np
from DFT import *
import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib.patches as mpatches
import scipy.special as spec
import matplotlib.animation as animation
from IPython.display import HTML



for i in range(60*3, 20*3, -10):
    f, ax = plt.subplots(3,1,sharex=False)
    camera = Camera(f)
    n = i# Number of data points
    T =  0.27*3# Sampling period
    dt = T/n
    vs = 1./dt
    tmin=0.
    tmax=dt*n
    print("sampling freq:",1./dt)
    nuc=0.5/dt

    t = dt*np.arange(0,n)
    t01 = 0.27
    print('najvisja_freq_signala:',1/t01*22)
    ht = 0.2*np.sin(2*np.pi*t/t01)  + 0.8*np.sin(2*np.pi*t/t01*2) + 0.2*np.sin(2*np.pi*t/t01*3) + \
    0.21*np.sin(2*np.pi*t/t01*4) + 0.63*np.sin(2*np.pi*t/t01*5) + np.sin(2*np.pi*t/t01*6) + \
    np.sin(2*np.pi*t/t01*7) + 0.57*np.sin(2*np.pi*t/t01*11) + np.sin(2*np.pi*t/t01*22)

    #ht = np.sin(2*np.pi * t/t01*22)
    nu= np.linspace(-nuc,nuc,n,endpoint=False)
    dft=DFT_slow(ht)
#dft=DFT_simplest(fx)
    Hk=np.roll(dft,int(n/2))/n
#fk=dft


    ax[0].set_xlim([0, tmax])
    ax[0].plot(t,ht, 'ro-', markersize = '2' )
    ax[0].set_xlabel(r't', size = 'x-large')
    ax[0].set_ylabel(r'h(t)', size = 'x-large')
    ax[2].set_xlim([-nuc, nuc])
    ax[1].set_xlim([-nuc, nuc])
    ax[2].set_ylabel(r'$\vert H_k \vert ^2$', size = 'x-large')
    ax[2].set_xlabel(r'$\nu$', size = 'x-large')
    ax[1].plot(nu, np.real(Hk),color='b')
    ax[1].set_ylabel(r'$Re[H_k]$', size = 'x-large')
    ax[2].plot(nu, np.imag(Hk),color='r')
    ax[2].set_ylabel(r'$Im[H_k]$', size = 'x-large')
    red_patch = mpatches.Patch(color='red', label=r'$T={}, \nu s = {}$'.format(np.round(T, 4),np.round(vs,2)))
    f.suptitle(r'$\nu_{max} = 81 \ Hz$')
    f.legend(handles=[red_patch])
    f.subplots_adjust(hspace=0.5)
    plt.show()
    camera.snap()


    #ims.append([im])
animation = camera.animate()
animation.save('potujevanje.gif', writer='Pillow', fps=1)
