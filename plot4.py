import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l_gw_sample = np.sort(dset['luminosity_distance_Mpc'])
d_l_gw = np.random.choice(d_l_gw_sample, size=d_l_gw_sample.size**2)
d_l = np.random.normal(loc=42.9, scale=3.2, size=d_l_gw_sample.size**2)
g_17 = (d_l_gw/d_l)**2

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

def plot17(lamdas=np.linspace(0, 42.9*3, 101)):
    js = np.empty(lamdas.size)
    for i in range(lamdas.size):
        print(i)
        lamda = lamdas[i]
        if (lamda == 0):
            e = np.zeros(10**8)
            a_17 = g_17-1
        else:
            e = (1+d_l/lamda)*np.exp(-d_l/lamda)
            a_17 = -((1-g_17)/(1-g_17*e))
        a_17 = a_17[np.where(abs(a_17)<=10)]
        bins = int((a_17.max()-a_17.min())*200)
        hist, bin_edges = np.histogram(a_17, bins=bins, density=True)
        a_17, p_a_17 = (bin_edges[:-1]+bin_edges[1:])/2, hist
        s = ((a_17[1:]-a_17[:-1])*(p_a_17[1:]+p_a_17[:-1])).sum()/2
        a_17, p_a_17 = a_17, p_a_17/s
        if (lamda == 0):
            e = 0
        else:
            e = (1+2.5e3/lamda)*np.exp(-2.5e3/lamda)
        alpha = -((1-g_19)/(1-g_19*e))
        p_alpha = p_g_19*((1-e)/(1+alpha*e)**2)
        alpha_i = np.sort_complex(alpha+p_alpha*1j)
        a_19, p_a_19 = alpha_i.real, alpha_i.imag
        from scipy.integrate import simpson
        a = np.linspace(a_17.min(), a_17.max(), 10**7)
        p_17 = np.interp(a, a_17, p_a_17, left=0, right=0)
        p_19 = np.interp(a, a_19, p_a_19, left=0, right=0)
        f = p_17*np.log2(p_17/((p_17+p_19)/2))
        f[np.isnan(f)] = 0
        kl_17 = simpson(f, a)
        a = np.linspace(a_19.min(), a_19.max(), 10**7)
        p_19 = np.interp(a, a_19, p_a_19, left=0, right=0)
        p_17 = np.interp(a, a_17, p_a_17, left=0, right=0)
        f = p_19*np.log2(p_19/((p_19+p_17)/2))
        f[np.isnan(f)] = 0
        kl_19 = simpson(f, a)
        js[i] = (kl_17+kl_19)/2
    plt.plot(lamdas, js)

def plot19(lamdas=np.linspace(2.5e3/3, 2.5e3*3, 101)):
    js = np.empty(lamdas.size)
    for i in range(lamdas.size):
        print(i)
        lamda = lamdas[i]
        if (lamda == 0):
            e = np.zeros(10**8)
            a_17 = g_17-1
        else:
            e = (1+d_l/lamda)*np.exp(-d_l/lamda)
            a_17 = -((1-g_17)/(1-g_17*e))
        a_17 = a_17[np.where(abs(a_17+1)<=0.5)]
        bins = int((a_17.max()-a_17.min())*200)
        hist, bin_edges = np.histogram(a_17, bins=bins, density=True)
        a_17, p_a_17 = (bin_edges[:-1]+bin_edges[1:])/2, hist
        s = ((a_17[1:]-a_17[:-1])*(p_a_17[1:]+p_a_17[:-1])).sum()/2
        a_17, p_a_17 = a_17, p_a_17/s
        if (lamda == 0):
            e = 0
        else:
            e = (1+2.5e3/lamda)*np.exp(-2.5e3/lamda)
        alpha = -((1-g_19)/(1-g_19*e))
        p_alpha = p_g_19*((1-e)/(1+alpha*e)**2)
        where = np.where(abs(a_17)<=10)
        alpha_i = np.sort_complex(alpha[where]+p_alpha[where]*1j)
        a_19, p_a_19 = alpha_i.real, alpha_i.imag
        s = ((a_19[1:]-a_19[:-1])*(p_a_19[1:]+p_a_19[:-1])).sum()/2
        a_19, p_a_19 = a_19, p_a_19/s
        from scipy.integrate import simpson
        a = np.linspace(a_17.min(), a_17.max(), 10**7)
        p_17 = np.interp(a, a_17, p_a_17, left=0, right=0)
        p_19 = np.interp(a, a_19, p_a_19, left=0, right=0)
        f = p_17*np.log2(p_17/((p_17+p_19)/2))
        f[np.isnan(f)] = 0
        kl_17 = simpson(f, a)
        a = np.linspace(a_19.min(), a_19.max(), 10**7)
        p_19 = np.interp(a, a_19, p_a_19, left=0, right=0)
        p_17 = np.interp(a, a_17, p_a_17, left=0, right=0)
        f = p_19*np.log2(p_19/((p_19+p_17)/2))
        f[np.isnan(f)] = 0
        kl_19 = simpson(f, a)
        js[i] = (kl_17+kl_19)/2
    plt.plot(lamdas, js)

plot17()
plot19()
plt.ylim()
plt.title('GW170817 vs GW190521')
plt.xlabel('$\\lambda$(Mpc)')
plt.ylabel('JS divergence')
plt.grid()
plt.savefig('plot4')
