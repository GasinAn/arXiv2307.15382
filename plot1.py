import h5py
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2)

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = np.sort(dset['luminosity_distance_Mpc'])
sample = np.sort((np.random.choice(d_l, size=10**8)
                 /np.random.normal(loc=42.9, scale=3.2, size=10**8))**2)
print(sample.mean(), sample.std())
print(sample[int(sample.size*(0.5-0.3413))],
      sample[int(sample.size*(0.5+0.3413))])
hist, bin_edges = np.histogram(sample, bins=300, density=True)
axs[0].plot((bin_edges[:-1]+bin_edges[1:])/2, hist, c='tab:blue')
axs[0].axvline(sample[int(10**8*0.0025)], c='tab:grey', ls='--')
axs[0].axvline(sample[int(10**8*0.9975)], c='tab:grey', ls='--')
axs[0].set_title('GW170817')
axs[0].set_xlim((-0.2,+2.2))
axs[0].set_ylim((-0.10,+1.85))
axs[0].set_xlabel('$G_s/G$')
axs[0].set_ylabel('$p(G_s/G)$')
axs[0].yaxis.set_label_position('left')
axs[0].yaxis.tick_left()
axs[0].grid()

img = mpimg.imread('d_l_GW190521.jpg')
sample = []
for x in range(500):
    ys = []
    for y in range(780):
        if (max(abs(img[y,x,0]-31),
                abs(img[y,x,1]-119),
                abs(img[y,x,2]-180)) <= 32):
            ys.append(y)
    if (ys != []):
        sample.append([x, np.array(ys).mean()])
x, y = np.array(sample)[:,0], np.array(sample)[:,1]
x, y = (x-983)/(240-983)*(2-10)+10, (y-9)/(699-9)*(0.0-0.7)+0.7
x, y = x/2.5, y*2.5
s = ((x[1:]-x[:-1])*(y[1:]+y[:-1])).sum()/2
x, y = x/s, y/s
s = 0
for i in range(len(x)):
    s += ((x[i+1]-x[i])*(y[i+1]+y[i]))/2
    if (s >= 0.0025):
        x_m = x[i+1]
        break
s = 0
for i in range(len(x)):
    s += ((x[-1-i]-x[-2-i])*(y[-1-i]+y[-2-i]))/2
    if (s >= 0.0025):
        x_p = x[-2-i]
        break
print(1.8/2.5, (1.8-0.1)/2.5, (1.8+1.1)/2.5)
axs[1].plot(x, y, c='tab:orange')
axs[1].axvline(x_m, c='tab:grey', ls='--')
axs[1].axvline(x_p, c='tab:grey', ls='--')
axs[1].set_title('GW190521')
axs[1].set_xlim((-0.2,+2.2))
axs[1].set_ylim((-0.10,+1.85))
axs[1].set_xlabel('$G_s/G$')
axs[1].set_ylabel('$p(G_s/G)$')
axs[1].yaxis.set_label_position('right')
axs[1].yaxis.tick_right()
axs[1].grid()

plt.show()
