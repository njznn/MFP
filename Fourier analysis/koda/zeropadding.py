import numpy as np
from DFT import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from timeit import default_timer as timer

n = 200 # Number of data points
T =  2# Sampling period
dt = T/n
vs = 1./dt
tmin=0.
tmax=dt*n
print("sampling freq:",1./dt)
nuc=0.5/dt

t = dt*np.arange(0,n)



nicle = 2*n
t = dt*np.arange(0,n)
t1 = np.append(t, np.arange(n+1,nicle+1)*dt)
#ht = np.sin(2*np.pi*t/0.25)
ht = np.sin(2*np.pi*t/0.23) + np.cos(2*np.pi*t/0.2)  +np.cos(2*np.pi*t/0.0444) + np.cos(2*np.pi*t/0.02633) + np.sin(2*np.pi*t/0.121679)
ht = np.append(ht, np.zeros(nicle-n))



plt.plot(t1,ht,'r.-')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title('h(t)')
red_patch = mpatches.Patch(color='red', label=r'$T={}, \nu s = {}$'.format(np.round(T, 4),np.round(vs,2)))
plt.legend(handles=[red_patch])
#h(t)=0.2s($\omega_1$t)+0.8s($2\omega_1$t) + 0.2s($3\omega_1$t)+\n0.21s($4\omega_1$t)+ 0.63s($5\omega_1$t)+s($6\omega_1$t) +\ns($7\omega_1$t) + 0.57s($11\omega_1$t) + s($22\omega_1$t)'
plt.show()
plt.clf()

nu= np.linspace(-nuc,nuc,len(t1),endpoint=False)


dft=DFT_slow(ht)
#dft=DFT_simplest(fx)

Hk=np.roll(dft,int(n))/n
#fk=dft

#ht_inv=np.roll(dft_inv,int(n/2))/n
#plt.cla()
f, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(nu, np.real(Hk),color='b')
ax[0].set_ylabel(r'$Re[H_k]$', size = 'x-large')
ax[1].plot(nu, np.imag(Hk),color='r')
ax[1].set_ylabel(r'$Im[H_k]$', size = 'x-large')
#ax[2].plot(t,np.imag(dft_inv) ,color='y')
#ax[2].set_ylabel(r'$\vert H_k \vert ^2$', size = 'x-large')
#ax[2].set_xlabel(r'$\nu$', size = 'x-large')
f.suptitle("FT[h(t)]")
plt.tight_layout()
plt.show()
plt.clf()
