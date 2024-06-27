from autocol import *
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt



uharica = np.loadtxt('/home/ziga/Desktop/FMF/3.letnik/MFP/FFT/uharica_cricki_potok.txt')
n = len(uharica) # Number of data points
dx = 1/44100 # Sampling period (in meters)
x = dx*np.arange(0,n) # x coordinates
w1 = 100.0 # sine wavelength (meters)
fx = uharica

#fx1 = 1./(np.sqrt(2.*np.pi)*5)*np.exp(-np.power((x - 0)/5, 2)/2) # signal
fx = fft.fftshift(fx)
Fk = fft.fft(fx)/n # Fourier coefficients (divided by n)
nu = fft.fftfreq(n,dx)
nu1 = fft.fftfreq(n*2-1,dx) # Natural frequencies
Fk = fft.fftshift(Fk) # Shift zero freq to center
nu = fft.fftshift(nu)
nu1 = fft.fftshift(nu1) # Shift zero freq to center
#ifk = fft.ifft(fft.fftshift(Fk))
cor = autocol(uharica)
#for i in range(n // 2):
#    cor[i + n//2] = 0
aFk = fft.fft(cor)/(2*n) # Fourier coefficients (divided by n)
#nu = fft.fftfreq(n,1/n) # Natural frequencies
aFk = fft.fftshift(aFk) # Shift zero freq to center




#plt.cla()
f, ax = plt.subplots(3,1,sharex=True)
# Plot Cosine terms
ax[0].plot(nu, np.real(Fk),color='b')
ax[0].set_ylabel(r'$Re[FFT]$', size = 'x-large')
ax[0].set_xlim([-500, 500])
# Plot Sine terms
ax[1].plot(nu, np.imag(Fk),color='r')
ax[1].set_ylabel(r'$Im[FFT]$', size = 'x-large')
# Plot spectral power
ax[2].plot(nu1, np.real(aFk),color='y')
ax[2].set_ylabel(r'$avtokoleracija$', size = 'x-large')
ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
ax[2].ticklabel_format(axis='y', style='sci', scilimits=(2,100))
f.suptitle("uharica-črički-potok")
f.tight_layout()
plt.show()
plt.clf()

#plt.plot(x, np.real(fft.fftshift(ifk)))
#plt.show()
