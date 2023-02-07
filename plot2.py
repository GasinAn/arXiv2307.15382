import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.subplots(layout='constrained')

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = np.sort(dset['luminosity_distance_Mpc'])
size = 10**8
sample = np.sort((np.random.choice(d_l, size=size)
                 /np.random.normal(loc=42.9, scale=3.2, size=size))**2)
bins = int(100*(sample.max()-sample.min()))
hist, bin_edges = np.histogram(sample, bins=bins, density=True)
g, p_g = (bin_edges[:-1]+bin_edges[1:])/2, hist
s = ((g[1:]-g[:-1])*(p_g[1:]+p_g[:-1])).sum()/2
g, p_g = g, p_g/s

lamdas, alphas = np.linspace(0, 3, 1001), np.linspace(-8, +8, 1001)
p_alphas = np.empty((alphas.size, lamdas.size))
for i in range(lamdas.size):
    lamda = lamdas[i]
    if (lamda == 0):
        e = 0
    else:
        e = (1+1/lamda)*np.exp(-1/lamda)
    alpha = -((1-g)/(1-g*e))
    p_alpha = p_g*((1-e)/(1+alpha*e)**2)
    alpha_i = np.sort_complex(alpha+p_alpha*1j)
    p_alphas[:,i] = np.interp(alphas, alpha_i.real, alpha_i.imag)
qcs = plt.contourf(lamdas, alphas, p_alphas, levels=1001)
lamdas = np.linspace(0, 3, 101)
alpha_mins, alpha_maxs = np.empty_like(lamdas), np.empty_like(lamdas)
for i in range(lamdas.size):
    lamda = lamdas[i]
    if (lamda == 0):
        e = 0
    else:
        e = (1+1/lamda)*np.exp(-1/lamda)
    alphas = np.sort(-((1-sample)/(1-sample*e)))
    alpha_mins[i] = alphas[int(size*(0+0.05/2))]
    alpha_maxs[i] = alphas[int(size*(1-0.05/2))]
plt.title('GW170817')
plt.xlabel('$\\lambda/D_L$')
plt.ylabel('$\\alpha$')
plt.colorbar(qcs, label='$p(\\alpha)$')
plt.plot(lamdas, alpha_mins, c='red')
plt.plot(lamdas, alpha_maxs, c='red')
plt.show()
