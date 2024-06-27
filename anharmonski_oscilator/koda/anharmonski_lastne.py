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

H = hamiltonka(3, 1, 100)
lastne, vektorji = LA.eigh(H)
PLOT_PROB = False
QPAD_FRAC = 0.4
N = lambda i: 1./np.sqrt(np.sqrt(np.pi)*2**i*factorial(i))
E = lastne[:10]
#print(E)
st_funkcij = 9
def psi(i, q):
    return(N(i)*spec.hermite(i)(q)*np.exp(-q*q/2.))

def psi_anhar(lastni_vektor,q):
    lastni = 0
    for i in range(len(lastni_vektor)):
        lastni += lastni_vektor[i]*psi(i,q)
    return(lastni)

def max_tocke(v):
    qmax = np.sqrt(2. * E[v]-3)
    return(-qmax, qmax)

def potencial(q):
    return (q**2 / 2 + 1*q**4)


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
    psi_v = psi_anhar(vektorji[:,i], q)
    E_v = E[i]
    plot_func(ax, psi_v, scaling=SCALING, yoffset=E_v)
    ax.text(s=r'${}\hbar\omega$'.format(E[i].round(2)), x=qmax-4.45,
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
ax.set_title(r'$[q]^{4}, \lambda = 1$', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#plt.savefig('sho-psi{}-{}.png'.format(PLOT_PROB+1, st_funkcij))
#plt.show()
############################dodatna################
H = H_0(100) - 5/2 * q2(100) + 1/10 * q4(100)
lastne, vektorji = LA.eigh(H)
PLOT_PROB = False
QPAD_FRAC = 5
N = lambda i: 1./np.sqrt(np.sqrt(np.pi)*2**i*factorial(i))
E = lastne[:10]
#print(E)
st_funkcij = 9

def potencial(q):
    return (q**2 / 2 - 5/2*q**2 + 1/10 * q**4)
def max_tocke(v):
    qmax = np.sqrt(2. * E[v])
    return(-qmax, qmax)

fig, ax = plt.subplots()
qmin, qmax = max_tocke(st_funkcij)
xmin, xmax = -5,5
q = np.linspace(-5, 5, 500)
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
    psi_v = psi_anhar(vektorji[:,i], q)
    E_v = E[i]
    plot_func(ax, psi_v, scaling=SCALING, yoffset=E_v)
    ax.text(s=r'${}\hbar\omega$'.format(E[i].round(2)), x=5,
            y=E_v, va='center')
    # Label the vibrational levels.

# The top of the plot, plus a bit.
ymax = E_v+0.9

ylabel = r'$\psi(q)$'
ax.text(s=ylabel, x=0, y=ymax, va='bottom', ha='center')
ax.set_xlabel('$q$')
ax.set_xlim(xmin, xmax)
ax.set_ylim(-11, ymax)
ax.spines['left'].set_position('center')
ax.set_yticks([])
ax.set_yscale('linear')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
print(lastne)
#plt.savefig('sho-psi{}-{}.png'.format(PLOT_PROB+1, st_funkcij))
plt.show()
