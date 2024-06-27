from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def autocol(seznam1):
    seznam1 = np.append(seznam1, np.zeros(len(seznam1)-1))
    Fk = fft.fft(seznam1)
    S = np.absolute(Fk*np.conjugate(Fk))
    rez = fft.fftshift(fft.ifft(S))/len(seznam1)
    return(rez)



uharica = np.loadtxt('/home/ziga/Desktop/FMF/3.letnik/MFP/FFT/uharica_cricki.txt')
if (len(uharica) % 2) == 0:
    n = len(uharica)
else:
    n = len(uharica)-1
    uharica = uharica[:-1]
# Number of data points
dx = 1/44100 # Sampling period (in meters)
x = dx*np.arange(0,2*n-1) # x coordinates # Sampling period (in meters)
x1 = dx*np.arange(0,n)
fx = fft.fftshift(uharica)
Fk = fft.fft(fx)/n # Fourier coefficients (divided by n)
nu = fft.fftfreq(2*n-1,dx) # Natural frequencies
Fk = fft.fftshift(Fk) # Shift zero freq to center
nu = fft.fftshift(nu) # Shift zero freq to center
cor = autocol(uharica)
#for i in range(n // 2):
#    cor[i + n//2] = 0
aFk = fft.fft(cor)/n # Fourier coefficients (divided by n)
#nu = fft.fftfreq(n,1/n) # Natural frequencies
aFk = fft.fftshift(aFk) # Shift zero freq to center

'''
f, ax = plt.subplots(2,1,sharex=False)
ax[0].plot(x1, uharica)
ax[1].set_xlabel('t')
#ax[0].set_xlim([0, 5])
ax[0].set_ylabel('signal')
'''

'''
ax[1].set_xlim([4,6])
ax[1].plot(x, cor, 'r-')
#ax[1].plot(x, signal.fftconvolve(uharica, uharica[::-1]/(2*n)), 'y-' )
#ax[1].plot(x, signal.correlate(uharica, uharica/(2.*n)), 'b-')
f.suptitle("uharica, črički, deroča reka-2")
ax[1].set_ylabel('avtokorelacija')
f.tight_layout()
plt.show()
plt.clf()
'''
prava = signal.correlate(uharica, uharica/(2.*n))

plt.plot(x, abs((cor-signal.correlate(uharica, uharica/(2.*n)))/prava), 'ro', markersize='2', label = 'implementirana-sig.cor()')
#plt.plot(x, abs((signal.fftconvolve(uharica, uharica[::-1]/(2*n))-prava)/prava), 'bo', markersize='2', label = 'fft_conv()-sig.cor()')
plt.xlabel('x')
plt.ylabel(r'$\Delta_f / f$')
plt.legend()
plt.show()

#############filter#####
#plt.xlim([-500,500])
#plt.plot(nu, np.real(aFk))
#plt.show()
