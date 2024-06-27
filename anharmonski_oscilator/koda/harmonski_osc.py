from metode_qr import *
from hamiltonke import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
from scipy.linalg import schur
from timeit import default_timer as timer
import scipy.linalg as lin
import scipy.special as spec
from scipy.special import factorial
import matplotlib.pyplot as plt

PLOT_PROB = False
QPAD_FRAC = 1.3
N = lambda i: 1./np.sqrt(np.sqrt(np.pi)*2**i*factorial(i))
E = lambda v: v + 0.5
st_funkcij = 10
def psi(i, q):
    return(N(i)*spec.hermite(i)(q)*np.exp(-q*q/2.))

def max_tocke(v):
    qmax = np.sqrt(2. * E(v + 0.5))
    return(-qmax, qmax)

def potencial(q):
    return q**2 / 2


fig, ax = plt.subplots()
qmin, qmax = max_tocke(st_funkcij)
xmin, xmax = QPAD_FRAC * qmin, QPAD_FRAC * qmax
q = np.linspace(qmin, qmax, 500)
V = potencial(q)

def plot_func(ax, f, scaling=1, yoffset=0):
    """Plot f*scaling with offset yoffset.

    The curve above the offset is filled with COLOUR1; the curve below is
    filled with COLOUR2.

    """
    ax.plot(q, f*scaling + yoffset)


ax.plot(q, V, color='k', linewidth=1.5)
SCALING = 0.7
for i in range(st_funkcij+1):
    psi_v = psi(i, q)
    E_v = E(i)
    plot_func(ax, psi_v, scaling=SCALING, yoffset=E_v)
    ax.text(s=r'$\frac{{{}}}\hbar\omega$'.format(E[i]), x=qmax+0.2,
            y=E_v, va='center')
    # Label the vibrational levels.

# The top of the plot, plus a bit.
ymax = E_v+0.9

ylabel = r'$\psi(q)$'
ax.text(s=ylabel, x=0, y=ymax, va='bottom', ha='center')

ax.set_xlabel('$q$')
ax.set_xlim(xmin, xmax)
ax.set_ylim(0, ymax)
ax.spines['left'].set_position('center')
ax.set_yticks([])
ax.set_yscale('linear')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#plt.savefig('sho-psi{}-{}.png'.format(PLOT_PROB+1, st_funkcij))
plt.show()
