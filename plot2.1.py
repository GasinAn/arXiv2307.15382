import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

'''
fig, axs = plt.subplots(ncols=2, layout='constrained')
'''

'''
e = (1+d_l/lamda)*exp(-d_l/lamda)
g = +((1+a)/(1+a*e))
a = -((1-g)/(1-g*e))
p_a = p_g*|dg/da| = p_g*|((1-e)/(1+a*e)**2)|
e = (1-g+a)/(g*a)
p_e = p_g*|dg/de| = p_g*|((1+a)*a/(1+a*e)**2)|
'''

f = h5py.File('GW170817_GWTC-1.hdf5','r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = np.sort(dset['luminosity_distance_Mpc'])
sample = (np.random.choice(d_l, size=10**8)
         /np.random.normal(loc=42.9, scale=3.2, size=10**8))**2
hist, bin_edges = np.histogram(sample, bins=200, density=True)
g, p_g = (bin_edges[:-1]+bin_edges[1:])/2, hist
s = ((g[1:]-g[:-1])*(p_g[1:]+p_g[:-1])).sum()/2
g, p_g = g, p_g/s

'''
def plot_17(lamdas=np.linspace(0, 3, 101),
            alphas=np.linspace(-1.5, 0.5, 1001)):
    axs[0].set_xlim((alphas.min(), alphas.max()))
    p_alphas = np.empty((lamdas.size, alphas.size))
    for i in range(lamdas.size):
        lamda = lamdas[i]
        if (lamda == 0):
            e = 0
        else:
            e = (1+1/lamda)*np.exp(-1/lamda)
        alpha = -((1-g)/(1-g*e))
        p_alpha = p_g*((1-e)/(1+alpha*e)**2)
        alpha_i = np.sort_complex(alpha+p_alpha*1j)
        p_alphas[i] = np.interp(alphas, alpha_i.real, alpha_i.imag)
    return p_alphas.max(), axs[0].contourf(alphas, lamdas, p_alphas, levels=101)

max_17, qcs_17 = plot_17()
axs[0].set_title('GW170817')
axs[0].set_xlabel('$\\alpha$')
axs[0].set_ylabel('$\\lambda/D_{L,17}$')
axs[0].yaxis.set_label_position('right')
axs[0].yaxis.tick_right()
'''


alphas = np.linspace(-0.8, +0.8, 100)
es = np.linspace(-1, +2, 1000)
plt.xlim((es.min(), es.max()))
p_es = np.empty((alphas.size, es.size))
for i in range(alphas.size):
    print(i)
    alpha = alphas[i]
    e = (1-sample+alpha)/(sample*alpha)
    bins = int(200*(e.max()-e.min()))
    hist, bin_edges = np.histogram(e, bins=bins, density=True)
    e, p_e = (bin_edges[:-1]+bin_edges[1:])/2, hist
    p_es[i] = np.interp(es, e, p_e, left=0, right=0)
qcs = plt.contourf(es, alphas, p_es, levels=100)
plt.xlabel('$(1+d_l/\\lambda)\\exp(-d_l/\\lambda)$')
plt.ylabel('$\\alpha$')
plt.colorbar(qcs, label='$p$')
plt.savefig('plot2_1')


'''
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
g, p_g = g, p_g/s

def plot_19(lamdas=np.linspace(0, 3, 101),
            alphas=np.linspace(-1.5, 0.5, 1001)):
    axs[1].set_xlim((alphas.min(), alphas.max()))
    p_alphas = np.empty((lamdas.size, alphas.size))
    for i in range(lamdas.size):
        lamda = lamdas[i]
        if (lamda == 0):
            e = 0
        else:
            e = (1+1/lamda)*np.exp(-1/lamda)
        alpha = -((1-g)/(1-g*e))
        p_alpha = p_g*((1-e)/(1+alpha*e)**2)
        alpha_i = np.sort_complex(alpha+p_alpha*1j)
        p_alphas[i] = np.interp(alphas, alpha_i.real, alpha_i.imag)
    return p_alphas.max(), axs[1].contourf(alphas, lamdas, p_alphas, levels=101)

max_19, qcs_19 = plot_19()
axs[1].set_title('GW190521')
axs[1].set_xlabel('$\\alpha$')
axs[1].set_ylabel('$\\lambda/D_{L,19}$')
axs[1].yaxis.set_label_position('left')
axs[1].yaxis.tick_left()
'''

'''
fig.colorbar(qcs_17, ax=axs[0], location='left' , label='$p(\\alpha)$',
             format='%3.1f')
fig.colorbar(qcs_19, ax=axs[1], location='right', label='$p(\\alpha)$',
             format='%3.1f')
fig.savefig('plot2_1')
'''
