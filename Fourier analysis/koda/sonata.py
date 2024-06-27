import numpy as np
from DFT import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from timeit import default_timer as timer
from scipy.fft import fft


bach_2756 = np.loadtxt('/home/ziga/Desktop/FMF/3.letnik/MFP/Fourierova_analiza/bach_2756.txt')
#print(bach_44100)
dt = 1/2756
n = len(bach_2756)
T = n*dt
t = dt* np.arange(0, n)
nuc=0.5/dt
nu= np.linspace(-nuc,nuc,n,endpoint=False)
dft=DFT_faster(bach_2756)
Hk=np.roll(dft,int(n/2))



plt.plot(t,bach_2756,'.-', markersize = '1.5')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title('2756')
plt.show()
plt.clf()

f, ax = plt.subplots(3,1,sharex=False)
ax[0].plot(nu, np.real(Hk),color='b')
ax[0].set_ylabel(r'$Re[H_k]$', size = 'x-large')
ax[1].plot(nu, np.imag(Hk),color='r')
ax[1].set_ylabel(r'$Im[H_k]$', size = 'x-large')
ax[2].plot(nu[int(n/2):], np.absolute(Hk[int(n/2):]),color='y')
ax[2].set_ylabel(r'$\vert H_k \vert$', size = 'x-large')
ax[2].set_xlabel(r'$\nu$', size = 'x-large')
f.suptitle("FT[2756]")
plt.tight_layout()
plt.show()
plt.clf()
