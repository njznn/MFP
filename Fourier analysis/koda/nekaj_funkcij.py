import numpy as np
from DFT import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from timeit import default_timer as timer

n = 1000 # Number of data points
T =  500# Sampling period
dt = T/n
vs = 1./dt
tmin=0.
tmax=dt*n
print("sampling freq:",1./dt)
nuc=0.5/dt

t = dt*np.arange(-n/2,n/2)
t01 = 0.27
#print("nu1=",1./t01)
#print("nu2 =",1./0.02633, 1/0.2, 1/0.23, 1/0.0444,1/0.121679 )
ht = 1./(np.sqrt(2.*np.pi)*5)*np.exp(-np.power((t - 0)/5, 2.)/2)

#ht = np.sin(2*np.pi*t/t01*22)
'''
ht = 0.2*np.sin(2*np.pi*t/t01)  + 0.8*np.sin(2*np.pi*t/t01*2) + 0.2*np.sin(2*np.pi*t/t01*3) + \
0.21*np.sin(2*np.pi*t/t01*4) + 0.63*np.sin(2*np.pi*t/t01*5) + np.sin(2*np.pi*t/t01*6) + \
np.sin(2*np.pi*t/t01*7) + 0.57*np.sin(2*np.pi*t/t01*11) + np.sin(2*np.pi*t/t01*22)
'''
#ht = np.sin(2*np.pi*t/0.23) + np.cos(2*np.pi*t/0.2)  +np.cos(2*np.pi*t/0.0444) + np.cos(2*np.pi*t/0.02633) + np.sin(2*np.pi*t/0.121679)
ht1 = ht
ht = np.roll(ht,int(n/2))
plt.plot(t,ht1,'r.-')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title('h(t)')
#red_patch = mpatches.Patch(color='red', label=r'$T={}, \nu s = {}$'.format(np.round(T, 4),np.round(vs,2)))
red_patch = mpatches.Patch(color='red', label=r'$\mu = 0, \sigma = 5$')

plt.legend(handles=[red_patch])
#h(t)=0.2s($\omega_1$t)+0.8s($2\omega_1$t) + 0.2s($3\omega_1$t)+\n0.21s($4\omega_1$t)+ 0.63s($5\omega_1$t)+s($6\omega_1$t) +\ns($7\omega_1$t) + 0.57s($11\omega_1$t) + s($22\omega_1$t)'
#plt.show()
plt.clf()

nu= np.linspace(-nuc,nuc,n,endpoint=False)


dft=DFT_slow(ht)
#dft=DFT_simplest(fx)
Hk=np.roll(dft,int(n/2))
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
#plt.show()
plt.clf()
#####################inverz##############

#realna in imaginarna se zamenjajo pri inverzni FT!
#ht_inv=np.roll(Hk,int(n/2))
Ht_inv = np.roll(Hk,int(n/2))/n
idft_simpl = DFT_simplest_inv(Ht_inv)
idft_slow = DFT_slow_inv(Ht_inv)
idft_slow1 = np.roll(idft_simpl,int(n/2))
plt.plot(t, np.real(idft_slow1))
plt.title('IFT[Hk(w)]')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.legend()
#plt.show()
plt.clf()

razlika_slow = abs(np.real(idft_slow) - ht)
razlika_simpl = abs(np.real(idft_simpl) - ht)
plt.plot(t, razlika_slow, label='skalarno')
plt.plot(t, razlika_simpl, label='vektorsko')
plt.xlabel("t")
plt.yscale('log')
plt.ylabel("|IFT[Hk(w)] - h(t)|")
plt.legend()
#plt.show()
plt.clf()

 #casovna zahtevnost
casi_slow = np.array([])
casi_simpl = np.array([])
casi_f = np.array([])
for i in range(10, 700, 10):
    n = i*3 # Number of data points
    T =  0.27*3# Sampling period
    dt = T/n
    vs = 1./dt
    t = dt*np.arange(0,n)
    tmin=0.
    tmax=dt*n
    nuc=0.5/dt
    ht = 0.2*np.sin(2*np.pi*t/t01)  + 0.8*np.sin(2*np.pi*t/t01*2) + 0.2*np.sin(2*np.pi*t/t01*3) + \
    0.21*np.sin(2*np.pi*t/t01*4) + 0.63*np.sin(2*np.pi*t/t01*5) + np.sin(2*np.pi*t/t01*6) + \
    np.sin(2*np.pi*t/t01*7) + 0.57*np.sin(2*np.pi*t/t01*11) + np.sin(2*np.pi*t/t01*22)
    casf = timer()
    dft2 = DFT_faster(ht)
    Hk=np.roll(dft,int(n/2))/n
    casfk = timer()
    casi_f = np.append(casi_f, casfk - casf)
    cslo_z = timer()
    dft=DFT_slow(ht)
    Hk=np.roll(dft,int(n/2))/n
    cslo_k = timer()
    casi_slow = np.append(casi_slow, cslo_k-cslo_z)
    csim_z = timer()
    #dft1=DFT_simplest(ht)
    #Hk=np.roll(dft1,int(n/2))/n
    csim_k = timer()
    casi_simpl = np.append(casi_simpl, csim_k-csim_z)
plt.plot([i for i in range(10,700,10)], casi_slow, 'ro-' ,markersize = '2', label='vektorsko')
#plt.plot([i for i in range(10,400,10)], casi_simpl, 'bo-', markersize = '2', label = 'skalarno')
plt.plot([i for i in range(10,700,10)], casi_f, 'yo-', markersize = '2', label = 'hibrid')
plt.xlabel('N')
plt.ylabel('t[s]')
plt.legend()

plt.show()
