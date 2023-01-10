import h5py
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = np.sort(dset['luminosity_distance_Mpc'])

sample = (np.random.choice(d_l, size=10**8)
         /np.random.normal(loc=42.9, scale=3.2, size=10**8))**2-1
print(np.log(1+sample.max()))

def plot_17(lamdas=[0]):
    # (1+alpha)/(1+alpha*exp(-d_l/lamda)) == 1+sample
    # alpha == sample/(1-exp(-d_l/lamda)-exp(-d_l/lamda)*sample)
    # alpha >= -1
    # 0 <= (1+alpha)/(1+alpha*exp(-d_l/lamda)) < 1/exp(-d_l/lamda)
    for lamda in lamdas:
        if (lamda == 0):
            alpha = sample
        else:
            e = np.exp(-42.9/lamda)
            alpha = sample/(1-e-e*sample)
        alpha.sort()
        print(alpha.mean(), alpha.std())
        print(alpha[int(alpha.size*(0.5-0.3413))],
              alpha[int(alpha.size*(0.5+0.3413))])
        print(alpha[int(alpha.size*(0.5-0.4987))],
              alpha[int(alpha.size*(0.5+0.4987))])
        hist, bin_edges = np.histogram(alpha, bins=200, 
                                       range=(-1.0, +1.5), density=True)
        plt.plot((bin_edges[:-1]+bin_edges[1:])/2, hist)
plot_17([0,42.9/3.0,42.9/1.5])
plt.show()

#img = mpimg.imread('d_l_GW190521.jpg')
#sample = []
#for x in range(500):
#    ys = []
#    for y in range(780):
#        if (max(abs(img[y,x,0]-31),
#                abs(img[y,x,1]-119),
#                abs(img[y,x,2]-180)) <= 32):
#            ys.append(y)
#    if (ys != []):
#        sample.append([x, np.array(ys).mean()])
#x, y = np.array(sample)[:,0], np.array(sample)[:,1]
#x, y = (x-100)/((426-100)/(4-0))/2.5, (y-699)/((699-11)/(0-0.7))*2.5
#print(1.8/2.5, (1.8-0.1)/2.5, (1.8+1.1)/2.5)
#plt.plot(x, y)
#plt.xlabel('${G_\\mathrm{s}/G}$')
#plt.ylabel('$p({G_\\mathrm{s}/G})$')
#plt.legend(('GW170817', 'GW190521'))
#plt.grid()
#plt.savefig('plot.jpg', dpi=800)
