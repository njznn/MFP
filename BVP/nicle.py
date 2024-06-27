import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import scipy.special as spec
import sys
from mpmath import mp
from decimal import*


def bisekcija(f,a,b, l):
  #http://sl.wikipedia.org/wiki/Bisekcija_(numerična_metoda)
  a = a * 1.0
  b = b * 1.0
  n=1   # stevilo korakov - štejemo korake, zato da lahko nastavimo največje število korakov (max_korak)
  epsilon = 10 ** -5 # natančnost rešitve
  max_korakov = 420 # da ne pretiravamo izberemo recimo 420
  ret = False # ret je kaj bo funkcija vrnila
  if f(a)*f(b) > 0:
      return None
  while  n <= max_korakov:
    ret = ( a + b ) / 2.0  #sredina intervala
    if abs(b-a) < epsilon:
        return(ret)
    if abs(f(ret)) < epsilon: # tukaj je epsilon zarad pretirane natacnosti
      return(ret)
    else:
      n = n + 1 #nasledni korak, premikamo meje
      if f(ret) * f(a) > 0: #pogledamo na katerem od intervalov [a,ret] ali [ret,b] nam spremeni predznak in glede na to metodo nadaljujemo na manjšem intervalu
        a=ret
      else:
        b=ret

  return(ret)

g = lambda x: spec.airy(x)[2]
f = lambda x: spec.airy(x)[0]
def generator(spodnja, zgornja, korak, g):
    nicle = np.array([])
    a = np.arange(spodnja,zgornja,korak)
    for i in range(len(a)-2):
        nicle = np.append(nicle, bisekcija(g,a[i],a[i+1]) if bisekcija(g,a[i],a[i+1]) != None else 0 )
    return(nicle)

def prave_nicle(seznam):
    nicle = np.array([])
    for i in seznam:
        if i !=0.:
            nicle = np.append(nicle, i)
        else:
            None
    return(nicle)
#print(prave_nicle(nicle_B), len(prave_nicle(nicle_B)))

if __name__ == "__main__":

    nicle_A=  prave_nicle(nicle_A[::-1])
    nicle_B = prave_nicle(nicle_B[::-1])
    ################################################ po formuli: ############3

    F =lambda x: x**(2./3.)*(1+ (5/48)*x**(-2) - (5/36)*x**(-4) + (77125/82944)*x**(-6)-(108056875/6967296)*x**(-8))
    Ai = np.array([])
    Bi = np.array([])
    for s in range(2,102): #najverjetneje napaka v indeksu, saj se ce je s=1 nicle ne poklopijo
        Bi = np.append(Bi, -F((3*np.pi*(4*s-3))/8))
    for s in range(1,101):
        Ai = np.append(Ai, -F((3*np.pi*(4*s-1))/8))
    ########################################################################


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_yscale('log')
    ax2.yaxis.set_label_position("right")
    ax1.plot(Ai[0:15],abs(Ai[0:15]-nicle_A[0:15]), 'bo', markersize = '3')
    ax2.plot(Ai[15:],abs(Ai[15:]-nicle_A[15:]), 'bo', markersize = '3')
    ax1.set(xlabel = 'x', ylabel = 'Log(|$N_{B} - N_{R}$|)' )
    ax2.yaxis.tick_right()
    ax2.set(xlabel = 'x', ylabel = '$|N_{B} - N_{R}$|' )
    #plt.show()
    plt.close()
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_yscale('log')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax1.plot(Bi[0:15],abs(Bi[0:15]-nicle_B[0:15]), 'bo', markersize = '3')
    ax2.plot(Bi[15:],abs(Bi[15:]-nicle_B[15:]), 'bo', markersize = '3')
    ax1.set(xlabel = 'x', ylabel = 'Log(|$N_{B} - N_{R}$|)' )
    ax2.set(xlabel = 'x', ylabel = '$|N_{B} - N_{R}$|' )
    plt.show()
    plt.close()




#plt.plot(Ai, abs(Ai-nicle_A[::-1]))
#plt.show()
