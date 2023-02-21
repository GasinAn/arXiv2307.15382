import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.subplots(layout='constrained')

def p_g(g):
    mu, sigma = 0.8, 0.3
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(g-mu)**2/(2*sigma**2))

def p_alpha(alpha, lamda):
    if (lamda == 0):
        e = 0
    else:
        e = (1+1/lamda)*np.exp(-1/lamda)
    g = (1+alpha)/(1+alpha*e)
    return p_g(g)*((1-e)/(1+alpha*e)**2)

lamdas = np.linspace(3, 0, 1001)
alphas = np.linspace(-4, +4, 10001)
from scipy.integrate import quad
for lamda in lamdas:
    e = (1+1/lamda)*np.exp(-1/lamda)
    if (quad(p_alpha, -np.inf, -1/e, args=tuple([lamda]))[0] >= 0.05/2):
        continue
    else:
        break
print(lamda)
plt.plot(alphas, p_alpha(alphas, lamda))
plt.show()
