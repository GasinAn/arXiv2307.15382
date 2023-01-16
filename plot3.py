import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

cmap = 'viridis'
colors = mpl.colormaps[cmap].resampled(1001)

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l_gw = np.sort(dset['luminosity_distance_Mpc'])

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
g, p_g = np.array(sample)[:,0], np.array(sample)[:,1]
g, p_g = (g-983)/(240-983)*(2-10)+10, (p_g-9)/(699-9)*(0.0-0.7)+0.7
g, p_g = g/2.5, p_g*2.5
s = ((g[1:]-g[:-1])*(p_g[1:]+p_g[:-1])).sum()/2
g_19, p_g_19 = g, p_g/s

def plot(lamdas=[0]):
    for lamda in lamdas:
        d_l = np.random.normal(loc=42.9, scale=3.2, size=10**8)
        g_17 = (np.random.choice(d_l_gw, size=10**8)/d_l)**2
        if (lamda == 0):
            e = np.zeros(10**8)
            a_17 = g_17-1
        else:
            e = (1+d_l/lamda)*np.exp(-d_l/lamda)
            a_17 = -((1-g_17)/(1-g_17*e))
        a_17 = a_17[np.where(abs(a_17)<=5)]
        bins = int((a_17.max()-a_17.min())*200)
        hist, bin_edges = np.histogram(a_17, bins=bins, density=True)
        a_17, p_a_17 = (bin_edges[:-1]+bin_edges[1:])/2, hist
        s = ((a_17[1:]-a_17[:-1])*(p_a_17[1:]+p_a_17[:-1])).sum()/2
        a_17, p_a_17 = a_17, p_a_17/s
        if (lamda == 0):
            e = 0
            a_19 = g_19-1
            p_a_19 = p_g_19
        else:
            e = (1+2.5e3/lamda)*np.exp(-2.5e3/lamda)
            a_19 = -((1-g_19)/(1-g_19*e))
            p_a_19 = p_g_19*((1-e)/(1+a_19*e)**2)
        s = ((a_19[1:]-a_19[:-1])*(p_a_19[1:]+p_a_19[:-1])).sum()/2
        a_19, p_a_19 = a_19, p_a_19/s
        a = np.sort(np.hstack((a_17, a_19)))
        p_a = (np.interp(a, a_17, p_a_17, left=0, right=0)
              +np.interp(a, a_19, p_a_19, left=0, right=0))/2
        #plt.plot(a_17, p_a_17)
        #plt.plot(a_19, p_a_19)
        if (lamda == 0):
            plt.plot(a, p_a,
                     c='black')
        else:
            plt.plot(a, p_a,
                     c=colors(np.log10(lamda/42.9)*(0.5/np.log10(4))+0.5))

plot()
plot(42.9*np.logspace(np.log10(1/3), np.log10(1*3), 7))
plt.xlim((-1.5, +0.5))
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
             label='$\\lambda$(Mpc)',
             ticks=42.9*np.logspace(np.log10(1/4), np.log10(1*4), 9),
             boundaries=42.9*np.logspace(np.log10(1/4), np.log10(1*4), 1001),
             values=np.linspace(0, 1, 1000))
plt.show()
