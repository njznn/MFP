import numpy as np
import matplotlib.pyplot as plt
from bvp import *
from nicle import *
from diffeq import rk4
from scipy.linalg import eigh
import matplotlib as mpl
import scipy.linalg as la
from scipy.integrate import solve_bvp
from tridiagonal import *
#######################SHOOTING METHOD#################################
#t-polovica intervala
t = np.linspace(0, 0.5, 1000)

def analiticna_resitev(t, n,):
    if n % 2 == 1:
        return(1*np.cos(n*np.pi*t))
    if n%2 == 0:
        return(1*np.sin(n*np.pi*t))

def res_l_sode(t, l):
    def se(x,t):
        return np.array([x[1], -l * x[0]])
    resitev = rk4(se, [np.sqrt(2), 0.0], t)
    w_1 = resitev[len(t)-1, 0]
    return(w_1)
def res_l_lihe(t,l):
    def se(x,t):
        return np.array([x[1], -l * x[0]])
    resitev = rk4(se, [0.0, 1.0], t)
    w_1 = resitev[len(t)-1, 0]
    return(w_1)


def bisekcija(f,t,a,b,):
  #http://sl.wikipedia.org/wiki/Bisekcija_(numerična_metoda)
  a = a * 1.0
  b = b * 1.0
  n=1   # stevilo korakov - štejemo korake, zato da lahko nastavimo največje število korakov (max_korak)
  epsilon = 10 ** -7 # natančnost rešitve
  max_korakov = 420 # da ne pretiravamo izberemo recimo 420
  ret = False # ret je kaj bo funkcija vrnila
  if f(t,a)*f(t,b) > 0:
      return None
  while  n <= max_korakov:
    ret = ( a + b ) / 2.0  #sredina intervala
    if abs(b-a) < epsilon:
        return(ret)
    if abs(f(t,ret)) < epsilon: # tukaj je epsilon zarad pretirane natacnosti
      return(ret)
    else:
      n = n + 1 #nasledni korak, premikamo meje
      if f(t,ret) * f(t,a) > 0: #pogledamo na katerem od intervalov [a,ret] ali [ret,b] nam spremeni predznak in glede na to metodo nadaljujemo na manjšem intervalu
        a=ret
      else:
        b=ret

  return(ret)

def shoot(t, min_energija, maks_energija,korak, f):
    a = np.arange(min_energija, maks_energija, korak)
    lastne_en = np.array([])
    for i in range(len(a)-2):
        res = bisekcija(f,t,a[i],a[i+1]) if bisekcija(f,t,a[i],a[i+1]) != None else 0
        if res > 0:
            lastne_en = np.append(lastne_en, res)

    return(lastne_en)
def lastne_sode(t, last_en,):
    def se(x,t):
        return np.array([x[1], -last_en * x[0]])
    resitev = rk4(se, [1.0, 0.0], t)
    return resitev[:,0]

def lastne_lihe(t, last_en,):
    def se(x,t):
        return np.array([x[1], -last_en * x[0]])
    resitev = rk4(se, [0.0, 1.0], t)
    return resitev[:,0]


############################# DIFERENČNA METODA ###########################
n = 2000
m = n-2
x = np.linspace(-0.5, 0.5 ,n)
h = x[1]-x[0]
T=-2*np.diag(np.ones(m))+1*np.diag(np.ones(m-1),1) \
                +1*np.diag(np.ones(m-1),-1)



E, lastni = la.eigh(-1/(h*h)*T)
w = np.zeros((n, m))
for i in range(len(E)):
    w[1:-1, i] = lastni[:,i]
    if i % 2 == 1:
        if w[int(n/2+10), i] < 0.0:
            w[:,i] = -w[:,i]
    if i%2 ==0:
        if w[int(n/2+10), i] < 0:
            w[:,i] = -w[:,i]

#plt.plot(x[:], w[:,0] * 1/np.max(w[:,0]), 'b-', markersize='1')
#plt.plot(x[:], w[:,1] * 1/np.max(w[:,1]), 'b-', markersize='1')

#plt.show()
#####################grafi#################
'''
#sode_en  = shoot(t, 1.0, 5000.0, 50.0, res_l_sode)
#lihe_en = shoot(t, 1.0, 5000.0, 50.0, res_l_lihe)
#print(sode_en, lihe_en, len(sode_en), len(lihe_en))
prave_en = [(i*np.pi)**2 for i in range(1, 10)]
sode_en = np.array([9.86960477 ,  88.82643974 , 246.74010968 , 483.61061907, 799.437953,
 1194.22214127, 1667.96314812 ,2220.66099739, 2852.31568909, 3562.92722321,
 4352.49562359])
lihe_en = np.array([39.47842216,  157.91366196 , 355.3056488 ,  631.65490723,  986.96038818,
 1421.22247314, 1934.44268799, 2526.61950684, 3197.75292969 ,3947.84143066,
 4776.88806152])



plt.rcParams["axes.formatter.limits"]  = (-3,2)

fig, ax = plt.subplots(3,2)

for i in range(1,5):
    ax[0][0].set_xlabel('x')
    ax[0][0].set_title(r'$\psi_{n}$')
    ax[0][0].plot(x, analiticna_resitev(x, i), label='n = {}'.format(i))
    ax[2][0].set_xlim([0, 0.5])
    ax[2][0].plot(x, abs(analiticna_resitev(x, i)-w[:,i-1]* 1/np.max(w[:,i-1])), label='n = {}'.format(i))
ax[2][1].plot(t, abs(analiticna_resitev(t, 1)-lastne_sode(t,sode_en[0])), label='n=1')
ax[2][1].plot(t, abs(analiticna_resitev(t, 3)-lastne_sode(t,sode_en[1])), label='n=3')
ax[2][1].plot(t, abs(analiticna_resitev(t, 5)-lastne_sode(t,sode_en[2])), label='n=5')
ax[2][1].plot(t,abs(analiticna_resitev(t, 2)-lastne_lihe(t,lihe_en[0])/np.max(lastne_lihe(t,lihe_en[0]))), label='n=2')
ax[2][1].plot(t, abs(analiticna_resitev(t,4)-lastne_lihe(t,lihe_en[1])/np.max(lastne_lihe(t,lihe_en[1]))), label='n=4')
ax[2][1].legend()
ax[2][1].set_title(r'$ \Delta: strelska \ metoda$', fontsize='10')
ax[2][1].set_ylabel(r'$|\psi-\psi_{prava}|$')

ax[2][0].legend()
ax[2][0].set_xlabel('x')
ax[2][0].set_ylabel(r'$|\psi-\psi_{prava}|$')
ax[2][0].set_title(r'$ \Delta: diferenčna \ metoda$', fontsize='10')
ax[0][0].legend()
ax[1][0].plot([i for i in range(1,10, 2)], abs(prave_en[::2] - sode_en[:5])/prave_en[::2], 'ro', label='strelska metoda')
ax[1][0].plot([i for i in range(2,9, 2)], abs(prave_en[1::2] - lihe_en[:4])/prave_en[1::2], 'ro')
ax[1][0].set_xlabel('n')
ax[1][1].set_xlabel('n')
ax[1][1].plot([i for i in range(1,10)], abs(prave_en-E[:9])/prave_en, 'bo' ,label='diferenčna metoda')
ax[1][0].ticklabel_format(style='sci')
ax[1][1].ticklabel_format(style='sci')
ax[2][1].set_xlabel('x')
ax[1][0].set_ylabel(r'$\frac{|E-E_{prava}|}{E_{prava}}$')
ax[1][1].set_ylabel(r'$\frac{|E-E_{prava}|}{E_{prava}}$')
ax[1][0].legend(loc='upper left')
ax[1][1].legend()
l=0
for i in range(1, 10,2):

    print(i, l)
    ax[0][1].plot(i,la.norm(analiticna_resitev(t,i)-lastne_sode(t, sode_en[l])) , 'bo')
    ax[0][1].plot(i+1,la.norm(analiticna_resitev(t, i+1)-lastne_lihe(t,lihe_en[l])/np.max(lastne_lihe(t, lihe_en[l]))),  'go')
    ax[0][1].plot(i,la.norm(analiticna_resitev(x,i)-w[:,i-1]* 1/np.max(w[:,i-1]))/2 , 'ro')
    ax[0][1].plot(i+1,la.norm(analiticna_resitev(x,i+1)-w[:,i]* 1/np.max(w[:,i]))/2 , 'ro')


    l +=1
ax[0][1].set_yscale('log')
ax[0][1].set_ylabel(r'$||\psi-\psi_{prava}||$')
ax[0][1].set_xlabel('n')
ax[0][1].legend(['strelska-sodi', 'strelska-lihi', 'diferenčna'])
plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig.suptitle(r'$\infty \ potencialna \ jama$')
plt.show()
'''
#####################################
'''
n = np.arange(200, 4000, 500)
q = 1

x, y = np.meshgrid(n,q)
def analiticna_resitev(n1, n,):
    t = np.linspace(-0.5, 0.5, n1)
    if n % 2 == 1:
        return(1*np.cos(n*np.pi*t))
    if n%2 == 0:
        return(1*np.sin(n*np.pi*t))

def diference(n, lastna_fja):
    m = n-2
    x = np.linspace(-0.5, 0.5, m+2)
    h = x[1]-x[0]
    T=-2*np.diag(np.ones(m))+1*np.diag(np.ones(m-1),1) \
                    +1*np.diag(np.ones(m-1),-1)

    E, lastni = la.eigh(-1/(h*h)*T)
    w = np.zeros((m+2, m))
    for i in range(len(E)):
        w[1:-1, i] = lastni[:,i]
        if i % 2 == 1:
            if w[int(n/2+10), i] < 0.0:
                w[:,i] = -w[:,i]
        if i%2 ==0:
            if w[int(n/2+10), i] < 0:
                w[:,i] = -w[:,i]
    return(w[:, lastna_fja]/1/np.max(w[:,lastna_fja]))




for j, i in enumerate(n):
    plt.plot(np.linspace(-0.5, 0.5, i), abs(analiticna_resitev(i, 3) - diference(i, 2)), label = 'n = {}'.format(i))
plt.yscale('log')

plt.legend()
plt.xlabel('x')
plt.title('kvantno stanje:3')
plt.ylabel(r'$||\psi-\psi_{prava}||$')
plt.show()
'''
