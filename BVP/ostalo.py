import numpy as np
import matplotlib.pyplot as plt
from bvp import *
from nicle import *
from diffeq import rk4
from scipy.linalg import eigh
import scipy.linalg as la
from scipy.integrate import solve_bvp
from tridiagonal import *
from pylab import *
from scipy.integrate import odeint
from scipy.optimize import brentq
#######################SHOOTING METHOD#################################


V0  = 100.0

def rk4( f1, f2, x0, t):
    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        if t[i] <= 0.5:
            h = t[i+1] - t[i]
            k1 = h * f1( x[i], t[i])
            k2 = h * f1( x[i] + 0.5 * k1, t[i] + 0.5 * h)
            k3 = h * f1( x[i] + 0.5 * k2, t[i] + 0.5 * h)
            k4 = h * f1( x[i] + k3, t[i+1])
            x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        else :
            h = t[i+1] - t[i]
            k1 = h * f2( x[i], t[i])
            k2 = h * f2( x[i] + 0.5 * k1, t[i] + 0.5 * h)
            k3 = h * f2( x[i] + 0.5 * k2, t[i] + 0.5 * h)
            k4 = h * f2( x[i] + k3, t[i+1])
            x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return x
def bisekcija(f,t,a,b):
  #http://sl.wikipedia.org/wiki/Bisekcija_(numerična_metoda)
  a = a * 1.0
  b = b * 1.0
  n=1   # stevilo korakov - štejemo korake, zato da lahko nastavimo največje število korakov (max_korak)
  epsilon = 10 ** -8 # natančnost rešitve
  max_korakov = 600 # da ne pretiravamo izberemo recimo 420
  ret = False # ret je kaj bo funkcija vrnila
  if f(t,a)[0] * f(t,b)[0] > 0:
      return None, None
  while  n <= max_korakov:
    ret = ( a + b ) / 2.0  #sredina intervala
    if abs(b-a) < epsilon:
        return(ret, f(t,ret)[1])
    if abs(f(t,ret)[0]) < epsilon: # tukaj je epsilon zarad pretirane natacnosti
        return(ret, f(t,ret)[1])

    else:
      n = n + 1 #nasledni korak, premikamo meje
      if f(t,ret)[0] * f(t,a)[0] > 0: #pogledamo na katerem od intervalov [a,ret] ali [ret,b] nam spremeni predznak in glede na to metodo nadaljujemo na manjšem intervalu
        a=ret
      else:
        b=ret
  if n == max_korakov:
    return None, None
  else:
    return(ret, f(t,ret)[1])

def res_l_sode(t,l):
    def f1(x,t):
        return np.array([x[1], -l * x[0]])
    def f2(x,t):
        return np.array([x[1], -(l-V0) * x[0]])
    resitev = rk4(f1, f2, [1.0, 0.0], t)
    w_1 = resitev[-1, 0]
    return(w_1, resitev[:,0])

def res_l_lihe(t,l):
    def f1(x,t):
        return np.array([x[1], -l * x[0]])
    def f2(x,t):
        return np.array([x[1], -(l-V0) * x[0]])
    resitev = rk4(f1, f2, [0.0, 1.0], t)
    w_1 = resitev[len(t)-1, 0]
    return(w_1, resitev[:,0])
def shoot(t, min_energija, maks_energija,korak, f):
    a = np.arange(min_energija, maks_energija, korak)
    lastne_en = np.array([])
    lastni_v = np.zeros(len(t))
    for i in range(len(a)-2):
        res , lastni = bisekcija(f,t,a[i],a[i+1]) if bisekcija(f,t,a[i],a[i+1])[0] != None else (0,0)

        if res > 0:
            lastne_en = np.append(lastne_en, res)
            lastni_v = np.vstack((lastni_v, lastni))

    return(lastne_en,lastni_v)
def res_sode(t,l):
    def f1(x,t):
        return np.array([x[1], -l * x[0]])
    def f2(x,t):
        return np.array([x[1], -(l-V0) * x[0]])
    resitev = rk4(f1, f2, [1.0, 0.0], t)
    return resitev[:,0]
def res_lihe(t,l):
    def f1(x,t):
        return np.array([x[1], -l * x[0]])
    def f2(x,t):
        return np.array([x[1], -(l-V0) * x[0]])
    resitev = rk4(f1, f2, [0.0, 1.0], t)
    return resitev[:,0]


#lastne = shoot(t, 0.1,100.0, 0.5, res_l_sode)
#lastne_lihe = shoot(t, 1.0,100.0, 0.5, res_l_lihe)
#plt.plot(t[:100], res_sode(t, lastne[0])[:100])
#plt.plot(t[:100], res_lihe(t, lastne_lihe[0])[:100])
#plt.plot(t[:100], res_lihe(t, lastne_lihe[1])[:100])
#plt.plot(t[:100], res_sode(t, lastne[1])[:100])
#plt.show()
#print(lastne, lastne_lihe)

################################################# ANALITICNA RESITEV ######################

def find_analytic_energies(en):
    """
    Calculates Energy values for the finite square well using analytical
    model (Griffiths, Introduction to Quantum Mechanics, page 62.)
    """
    Vo = 100.0
    z = sqrt(en)
    z0 = sqrt(Vo)
    z_zeroes = []
    f_sym = lambda z: np.tan(z/2)-np.sqrt((z0/z)**2-1)      # Formula 2.138, symmetrical case
    f_asym = lambda z: 1/np.tan(z/2)+np.sqrt((z0/z)**2-1)  # Formula 2.138, antisymmetrical case

    # first find the zeroes for the symmetrical case
    s = sign(f_sym(z))
    for i in range(len(s)-1):   # find zeroes of this crazy function
       if s[i]+s[i+1] == 0:
           zero = brentq(f_sym, z[i], z[i+1])
           z_zeroes.append(zero)
    print ("Energies from the analytical model are: ")
    print ("Symmetrical case)")
    for i in range(0, len(z_zeroes),2):   # discard z=(2n-1)pi/2 solutions cause that's where tan(z) is discontinuous
        print ("%.4f" %(z_zeroes[i]**2))
    # Now for the asymmetrical
    z_zeroes = []
    s = sign(f_asym(z))
    for i in range(len(s)-1):   # find zeroes of this crazy function
       if s[i]+s[i+1] == 0:
           zero = brentq(f_asym, z[i], z[i+1])
           z_zeroes.append(zero)
    print ("(Antisymmetrical case)")
    for i in range(0, len(z_zeroes),2):   # discard z=npi solutions cause that's where cot(z) is discontinuous
        print ("%.4f" %(z_zeroes[i]**2))
en = np.arange(0.0, 100.0, 1.0)
#plt.plot(t[:500], res_sode(t, 5.9422)[:500])
#plt.plot(t[:500], res_lihe(t, 22.9768)[:500])
#plt.show()
################################KONCNE DIFERENCE################
def diferencna(N=5000):
    m = N-2
    x = np.linspace(-2.0, 2.0 ,N)
    h = x[1]-x[0]
    def matrika(x):
        m = N-2
        H = np.zeros([m, m])
        for i in range(m):
            H[i, i] = -2
            if(i-1>=0):
                H[i, i-1] = 1
            if(i+1<m):
                H[i, i+1] = 1
        return H
    V=np.zeros((m,m))
    for i in range(m):
        if x[i] < -0.5 or x[i] > 0.5:
            V[i,i] = 500
        else:
            V[i,i] = 0.0


    H = -1/(h*h) * matrika(x) + V
    E, lastni = eigh(H)
    w = np.zeros((N, m))
    for i in range(m):
        w[1:-1, i] = lastni[:,i]
        if i % 2 == 1:
            if w[int(N/2+10), i] < 0.0:
                w[:,i] = -w[:,i]
        if i%2 ==0:
            if w[int(N/2+10), i] < 0:
                w[:,i] = -w[:,i]
    return(E, w)
####################################grafi#############################3
E, w = diferencna(2000)

#print(E[:5])
E_dif = np.array([-93.17475662, -73.05589779, -41.11051238,  -3.70636533 ])
E_dif +=100
x =np.linspace(-2.0, 2.0, 2000)
####ENERGIJE ZA .V0=100###########
#E_dif = np.array([6.8168018 ,  26.9046807  , 58.7885011,  96.15227794])
#print(E_dif)
#print(find_analytic_energies(en))
analit_sode = np.array([6.8271, 58.9046])
analit_lihe = np.array([26.9514, 96.2869])
t = np.linspace(0.0,2.0 ,3000)
#print(shoot(t, 1.0, 100.0, 1.0, res_l_lihe))
#lastne_en, lastni = shoot(t, 1.0, 120.0, 5.0, res_l_sode)
lihe_en = np.array([26.93688596 , 96.27784288])
sode_en = np.array([6.82337204 , 58.87468385  ])
V0 = 500
for i in range(8):

    plt.plot(x, w[:,i]/(np.max(w[:,i])) + E[i]/50, label=r'$\psi_{}$'.format(i+1))
plt.plot(np.linspace(-2.0, -0.5, 100), 10*np.ones(100), c='k',ls='--', label='V0/50')
plt.plot(np.linspace(0.5,2.0, 100), 10*np.ones(100), c='k',ls='--')
plt.plot(np.linspace(-0.5, 0.5, 100), 0*np.ones(100), c='k',ls='--')
plt.axvline(x = 0.5,ymin = 0.04347826, ymax = 0.91, c='k',ls='--', label = '')
plt.axvline(x = -0.5,ymin = 0.04347826, ymax = 0.91, c='k',ls='--', )
plt.xlabel('x')
plt.ylabel(r'$\frac{10 \ E}{V0}$')
plt.ylim((-0.5,11))
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.legend()
#plt.plot(t, w[1000:, 1]/np.max(w[1000:, 1]))
#plt.plot(t, w[1000:, 3]/np.max(w[1000:, 3]))
plt.show()
