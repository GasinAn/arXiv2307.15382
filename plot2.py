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
plt.title('GW170817')
plt.xlabel('$\\alpha$')
plt.ylabel('$p(\\alpha)$')
plt.legend(('$\\lambda=0$','$\\lambda=d_L/3.0$','$\\lambda=d_L/1.5$'))
plt.grid()
plt.show()

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
x, p_x = x-1, y
print(np.log(1+x.max()))

def plot_19(lamdas=[0]):
    # alpha(x) == x/(1-e-e*x)
    # P(x<=?) == \int_-oo^? p_x(x) d{x}
    # P(alpha(x)<=?) == P(x<=alpha^-1(?)) == \int_-oo^alpha^-1(?) p_x(x) d{x}
    # x == alpha^-1(?) <=> alpha(x) == ?
    #    \int_-oo^alpha^-1(?) p_x(x) d{x}
    # == \int_-oo^alpha^-1(?) p_x(alpha^-1(alpha)) d{alpha^-1(alpha)}
    # == \int_-oo^?           p_x(alpha^-1(alpha))alpha^-1'(alpha) d{alpha}
    # alpha^-1(alpha) == (1+alpha)/(1+e*alpha)-1 == x
    # alpha^-1'(alpha) == (1-e)/(1+e*alpha)**2
    # p_alpha(alpha) = p_x((1+alpha)/(1+e*alpha)-1)*(1-e)/(1+e*alpha)**2
    #    \int alpha*p_alpha(alpha) d{alpha}
    # == \int alpha*p_x(alpha^-1) d{alpha^-1}
    # == \int alpha(x)*p_x(x) d{x} approx alpha(\int x*p_x(x) d{x}) (linear)
    for lamda in lamdas:
        if (lamda == 0):
            e = 0
            alpha, p_alpha = x, p_x
        else:
            e = np.exp(-2.5/lamda)
            alpha, p_alpha = x/(1-e-e*x), p_x*(1-e)/(1+e*alpha)**2
        where = np.where(alpha<=1.5)
        print(((e*x**2/(1-e)**2)/(x/(1-e))).mean())
        print(((e*x**2/(1-e)**2)/(x/(1-e))).std())
        print(((e*x**2/(1-e)**2)/(x/(1-e))).max())
        x_0, x_m, x_p = 1.8/2.5-1, (1.8-0.1)/2.5-1, (1.8+1.1)/2.5-1
        print(x_0/(1-e-e*x_0), x_m/(1-e-e*x_m), x_p/(1-e-e*x_p))
        plt.plot(alpha[where], p_alpha[where])
plot_19([0,2.5/2.0,2.5/1.0])
plt.title('GW190521')
plt.xlabel('$\\alpha$')
plt.ylabel('$p(\\alpha)$')
plt.legend(('$\\lambda=0$','$\\lambda=d_L/2.0$','$\\lambda=d_L/1.0$'))
plt.grid()
plt.show()
