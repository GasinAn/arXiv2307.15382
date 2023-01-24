import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2)

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

def plot(ax, lamdas,
         alphas=np.linspace(-1.5, 0.5, 1001)):
    plt.xlim((alphas.min(), alphas.max()))
    p_alphas = np.empty((lamdas.size, alphas.size))
    for i in range(lamdas.size):
        lamda = lamdas[i]
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
        p_alphas[i] = np.interp(alphas, a, p_a)
    return axs[ax].contourf(alphas, lamdas, p_alphas, levels=100)

qcs0 = plot(ax=0, lamdas=np.linspace(0, 42.9*3, 101))
qcs1 = plot(ax=1, lamdas=np.linspace(2.5e3/3, 2.5e3*3, 101))
for ax in axs:
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\lambda$(Mpc)')
for qcs in (qcs0, qcs1):
    plt.colorbar(qcs, label='$p(\\alpha)$')

fig.suptitle('GW170817+GW190521')
fig.savefig('plot3_1')
