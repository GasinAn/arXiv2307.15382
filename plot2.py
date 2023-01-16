import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#f = h5py.File('GW170817_GWTC-1.hdf5','r')
#dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
#d_l = np.sort(dset['luminosity_distance_Mpc'])
#sample = (np.random.choice(d_l, size=10**8)
#         /np.random.normal(loc=42.9, scale=3.2, size=10**8))**2-1
#print(np.log(1+sample.max()))

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
#plot_17([0,42.9/3.0,42.9/1.5])
#plt.title('GW170817')
#plt.xlabel('$\\alpha$')
#plt.ylabel('$p(\\alpha)$')
#plt.legend(('$\\lambda=0$','$\\lambda=d_L/3.0$','$\\lambda=d_L/1.5$'))
#plt.grid()
#plt.show()

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
g, p_g = g/s, p_g/s
print(g.max())

def plot_19(lamdas=[0]):
    # e := (1+d/l)*exp(-d/l)
    # (1+alpha)/(1+alpha*e) = g
    # alpha(g) = -((1-g)/(1-g*e))
    # P(g \in A) = \int_A p_g(g) d{g}
    # P(g \in alpha^{-1}(A)) = \int_alpha^{-1}(A) p_g(g) d{g}
    # P(alpha \in A) = \int_A p_alpha(alpha) d{alpha}
    #    \int_alpha^{-1}(A) p_g(g) d{g}
    # == \int_alpha^{-1}(A) p_g(alpha^-1(alpha)) d{alpha^-1(alpha)}
    # == \int_A             p_g@alpha^-1(alpha)*alpha^-1'(alpha) d{alpha}
    # p_alpha(alpha) = p_g@alpha^-1(alpha)*alpha^-1'(alpha)
    # alpha^-1(alpha) = (1+alpha)/(1+alpha*e) = g
    # alpha^-1'(alpha) = (1-e)/(1+alpha*e)**2
    # alpha(g) = -((g-1)/(g*e-1))
    # p_alpha(alpha) = p_g@alpha^-1(alpha)*((1-e)/(1+alpha*e)**2)
    for lamda in lamdas:
        if (lamda == 0):
            e = 0
            print(np.inf)
            alpha = g-1
            p_alpha = p_g
        else:
            e = (1+2.5/lamda)*np.exp(-2.5/lamda)
            print(1/e)
            alpha = -((1-g)/(1-g*e))
            p_alpha = p_g*((1-e)/(1+alpha*e)**2)
        #def F(g_):
        #    return -((1-g_)/(1-g_*e))*np.interp(g_, g, p_g)
        #from scipy.integrate import quad
        #print(quad(F, g.min(), g.max())[0])
        #g_0, g_m, g_p = 1.8/2.5, (1.8-0.1)/2.5, (1.8+1.1)/2.5
        #print(-((1-g_0)/(1-g_0*e)), -((1-g_m)/(1-g_m*e)), -((1-g_p)/(1-g_p*e)))
        alpha_i = np.sort_complex(alpha+p_alpha*1j)
        if (lamda == 0):
            plt.plot(alpha_i.real, alpha_i.imag,
                     c='black')
        else:
            plt.plot(alpha_i.real, alpha_i.imag,
                     c=colors(np.log10(lamda/2.5)*(0.5/np.log10(4))+0.5))
cmap = 'viridis'
colors = mpl.colormaps[cmap].resampled(1001)
plot_19()
plot_19(2.5*np.logspace(np.log10(1/3), np.log10(1*3), 9))
plt.xlim((-1.5, +0.5))
plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap),
             label='$\\lambda$(Gpc)',
             ticks=2.5*np.logspace(np.log10(1/4), np.log10(1*4), 9),
             boundaries=2.5*np.logspace(np.log10(1/4), np.log10(1*4), 1001),
             values=np.linspace(0, 1, 1000))
plt.title('GW190521')
plt.xlabel('$\\alpha$')
plt.ylabel('$p(\\alpha)$')
plt.grid()
plt.show()
