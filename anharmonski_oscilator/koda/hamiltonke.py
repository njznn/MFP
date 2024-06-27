import numpy as np
import time
import scipy.special as spec

def H_0(velikost):
    return(np.diag(np.array([i+1/2 for i in range(velikost)])))
#############<i|q|j>###########

def q(velikost):
    a = np.array([1/2*np.sqrt(i+(i-1)+1) for i in range(1,velikost)])
    return(np.diag(a, -1)+ np.diag(np.zeros(velikost,))+ np.diag(a, 1))

#a = q(10)
#b = a@a@a@a + H_0(10)
#for vrstica in b:
#    print('  '.join(map(str, vrstica)))
##########<i|q**2|j>###############
def q2(velikost):
    d = np.array([1/2*(2*j+1) for j in range(velikost)])
    zgornja = np.array([1/2*np.sqrt(j*(j-1)) for j in range(2, velikost)])
    spodnja = np.array([1/2*np.sqrt((j+1)*(j+2)) for j in range(0, velikost-2)])
    return(np.diag(d) + np.diag(zgornja, 2) + np.diag(spodnja, -2))

#print(q2(10)@q2(10))

##########<i|q**4|j>###############
def q4(velikost):
    najnizje = np.array([(1/2**2)*np.sqrt((j+4)*(j+3)*(j+2)*(j+1)) for j in range(0, velikost-4)])
    spodnja = np.array([(1/2**3) * np.sqrt((j+2)*(j+1))*4*(2*j + 3) for j in range(0, velikost-2)])
    diagonala = np.array([(1/2**4)*12*(2*j**2 + 2*j + 1) for j in range(velikost)])
    zgornja = np.array([(1/2**5) * np.sqrt(1/(j*(j-1)))*16*j*(2*j**2 - 3*j + 1) for j in range(2, velikost)])
    najvisja = np.array([1/2**6 * np.sqrt(1/(j*(j-1)*(j-2)*(j-3)))*16*j*(j**3 -6*j**2 + 11*j - 6) for j in range(4, velikost)])
    return(np.diag(najnizje, -4) + np.diag(spodnja, -2) + np.diag(diagonala) + np.diag(zgornja, 2) + np.diag(najvisja, 4))

def hamiltonka(metoda, faktor, velikost):
    if metoda == 1:
        a = q(velikost)
        return(faktor*(a@a@a@a) + H_0(velikost))
    elif metoda == 2:
        b = q2(velikost)
        return(faktor*(np.dot(b,b)) + H_0(velikost))
    elif metoda ==3:
        return(faktor*q4(velikost) + H_0(velikost))
    else:
        print('izbiraš lahko samo med številkami od 1 do 3')
