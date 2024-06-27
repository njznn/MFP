import numpy as np
from numpy import fft
from diffeq_dodane import *
from tridiagonal import *
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from B_zlepki import *
from fourier import *

#interval :
N = 1000
M = 1000
#konstante:
a = 1.
D=0.05
cas = 1.0

tk = np.linspace(0., cas, M)
xk = np.linspace(0., a, 500)
dx = a/N
dt = cas/M
fv = 1/dx
print(fv)
nuc = 0.5/dx
#####################

#H = res(500,500,1.0,1.0, 0.3)
H = res_periodicni()




fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xk = np.linspace(0,a, len(H[0, :]))
tk = np.linspace(0,cas, len(H[:, 0]))
X,Y = np.meshgrid(xk,tk)
surf = ax.plot_surface(X, Y, H,  cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#surf = ax.plot_surface(X, Y, abs(H-H_1), cmap='viridis',
#                       linewidth=0, antialiased=False)

#ax.plot_surface(X, Y, H_1, cmap='viridis',
#                       linewidth=0, antialiased=False)
ax.set_zticks([])
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title(r'$T(x, t)$')
#ax.text(0.8, 0.9 ,0.8, r'$D = 0.3$' +"\n" ' $\sigma = 0.2 $')
ax.text(0.8, 0.9 ,0.8, r'$D = 0.03$')
ax.text(0.8, 0.9 ,0.6, r'$T(x,0) = sin(4\pi x)$')

fig.colorbar(surf, shrink=0.5, aspect=20, label=r'$Temperatura$')
fig.tight_layout()
plt.show()
plt.clf()
'''
 #(pri N=1000, M=100)
B = H_1[::10, ::10]
print(np.shape(B), np.shape(H))
for i in range(7):
    plt.plot(xk, abs(H[i, :]-B[i, :]),label='t{}'.format(i))
plt.legend()
plt.yscale('log')
plt.ylabel('$|T_{fft} - T_{Bx}|$')
plt.xlabel('x')

plt.show()
'''
