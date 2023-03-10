import random
import numpy as np
import matplotlib.pyplot as plt

def bv(mu,sig, n, t):
    vel = np.zeros((n,t,3))
    #vel[:,0,3] = np.arange(0,t,1)
    vel[:,0,-1] = np.random.normal(mu, sig, n)
    return vel

def tau(n):
    return np.random.exponential(2.5e-3,n)

def vert(n, tau, t):
    vert = np.zeros((n,t))
    vert[:,:] -= tau
    print(vert)


n = 100
t = 100

x=bv(2000, 50, n, t)
#print(x)

# Copied----------------------------------
count, bins, ignored = plt.hist(x[:,0,-1], 100, density=True)

plt.plot(bins, 1/(50 * np.sqrt(2 * np.pi)) *

               np.exp( - (bins - 2000)**2 / (2 * 50**2) ),

         linewidth=2, color='r')

plt.show()

sample = tau(n)

bin = np.arange(0,15e-3,0.0001)

plt.hist(sample, bins=bin, edgecolor='blue') 
plt.title("Exponential Distribution") 
plt.show()
#-----------------------------------------

vert(n, sample, t)