import h5py
import numpy as np
from matplotlib import pyplot as plt

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = np.sort(dset['luminosity_distance_Mpc'])

# plt.plot(d_l, (np.arange(d_l.size)+1)/d_l.size)
# plt.show()
# plt.hist(d_l, bins=int(d_l.size**(1/2)), density=True)
# plt.show()

# plt.hist((np.random.choice(d_l, size=d_l.size**2)
#          /np.random.normal(loc=42.9, scale=3.2, size=d_l.size**2)),
#          bins=d_l.size, range=(0.25, 1.50), density=True, histtype='step')
# plt.xlabel('$\\sqrt{G_\\mathrm{s}/G}$')
# plt.show()
sample = np.sort((np.random.choice(d_l, size=d_l.size**2)
                 /np.random.normal(loc=42.9, scale=3.2, size=d_l.size**2))**2)
print(sample.mean(), sample.std())
print(sample[int(sample.size*(0.5-0.3413))],
      sample[int(sample.size*(0.5+0.3413))])
plt.hist(sample, bins=d_l.size, range=(0.0, 2.0), density=True, histtype='step')
plt.xlabel('${G_\\mathrm{s}/G}$')
plt.ylabel('$p({G_\\mathrm{s}/G})$')
plt.grid()
plt.show()
# plt.savefig('plot', dpi=800)
