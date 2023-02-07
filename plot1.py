import h5py
import numpy as np
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
plt.plot((bin_edges[:-1]+bin_edges[1:])/2, hist)
plt.axvline(sample[int(size*(0+0.05/2))], c='tab:grey', ls='--')
plt.axvline(sample[int(size*(1-0.05/2))], c='tab:grey', ls='--')
plt.title('GW170817')
plt.xlim((-0.1, +2.1))
plt.ylim((-0.10, +1.35))
plt.xlabel('$G_s/G$')
plt.ylabel('$p(G_s/G)$')
plt.grid()
plt.show()
