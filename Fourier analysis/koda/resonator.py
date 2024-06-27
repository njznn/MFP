import numpy as np
from DFT import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from timeit import default_timer as timer
from scipy.fft import fft

res_1 = np.loadtxt('/home/ziga/Desktop/FMF/3.letnik/MFP/Fourierova_analiza/poskus1_akres_novi.dat')
#print(bach_44100)
dt = 1/44100
n = len(res_1)
T = n*dt
t = dt* np.arange(0, n)
nuc=0.5/dt
nu= np.linspace(-nuc,nuc,n,endpoint=False)
dft=fft(res_1)
Hk=np.roll(dft,int(n/2))



plt.plot(t,res_1,'.-', markersize = '1.5')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title('predmet v resonatorju')
plt.show()
plt.clf()

f, ax = plt.subplots(3,1,sharex=False)
ax[0].plot(nu, np.real(Hk),color='b')
ax[0].set_ylabel(r'$Re[H_k]$', size = 'x-large')
ax[0].set_xlim([0, 20])
ax[1].plot(nu, np.imag(Hk),color='r')
ax[1].set_xlim([-200, 200])
ax[1].set_ylabel(r'$Im[H_k]$', size = 'x-large')
ax[2].set_xlim([0, 200])
ax[2].plot(nu[int(n/2):], np.absolute(Hk[int(n/2):]),color='y')
ax[2].set_ylabel(r'$\vert H_k \vert$', size = 'x-large')
ax[2].set_xlabel(r'$\nu$', size = 'x-large')
f.suptitle("FT[resonator 1]")
plt.tight_layout()
plt.show()
plt.clf()
