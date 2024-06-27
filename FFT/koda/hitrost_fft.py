from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from timeit import default_timer as timer
import math
from scipy.optimize import curve_fit


def casovna_zaht(max_tock):
    casi = np.array([])
    for i in range(1,max_tock, 10000):
        dx = 1 # Sampling period (in meters)
        x = dx*np.arange(0,i) # x coordinates
        w1 = 100.0 # sine wavelength (meters)


        fx1 = 1./(np.sqrt(2.*np.pi)*5)*np.exp(-np.power((x - 0)/5, 2)/2) # signal
        zac = timer()
        fx = fft.fftshift(fx1)
        Fk = fft.fft(fx)/i # Fourier coefficients (divided by n)
        nu = fft.fftfreq(i,dx) # Natural frequencies
        Fk = fft.ifftshift(Fk) # Shift zero freq to center
        nu = fft.fftshift(nu) # Shift zero freq to center
        kon = timer()
        casi = np.append(casi, kon-zac)
    return(casi)

casi = casovna_zaht(1000000)
f = lambda a,x: a*(x/2) * math.log(x, 2.0)
#plt.yscale('log', base=2).
plt.yscale('log', base=2)
N = np.array([i for i in range(1,1000000,10000)])
popt, pcov = curve_fit(f, N, casi)
plt.ylabel('log2(t)')
plt.xlabel('N')
plt.plot(N, casi, 'bo', markersize = '2')
plt.plot(N, f(N, *popt), label=r'$1.00000061 \ (N/2)log2(N))$')
plt.title('časovna zahtevnost FFT')
plt.legend(loc = 4)
plt.show()
print(popt)
