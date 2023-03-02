import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.subplots(layout='constrained')

from astropy.units import Mpc, m
Mpc2m = ((1*Mpc).to(m)).value

f = h5py.File('GW170817_GWTC-1.hdf5', 'r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l_gw = dset['luminosity_distance_Mpc']
size = d_l_gw.size**2
d_l_samples = np.random.normal(loc=42.9*Mpc2m, scale=3.2*Mpc2m, size=size)
g_samples = (np.random.choice(d_l_gw*Mpc2m, size=size)/d_l_samples)**2

from scipy.optimize import brentq
def dpap(alpha, p):
    return np.interp(alpha, a, p_a, left=0, right=0)-p
lamdas, alphas = np.linspace(0, 7e23, 1001), np.linspace(-5, +5, 5001)
p_alphas = np.empty((alphas.size, lamdas.size))
a_min, a_max = np.empty(lamdas.size), np.empty(lamdas.size)
for i in range(lamdas.size):
    print(i)
    l = lamdas[i]
    if (l == 0):
        e_samples = np.zeros_like(d_l_samples)
    else:
        e_samples = np.array((1+d_l_samples/l)*np.exp(-d_l_samples/l))
    a_samples = -((1-g_samples)/(1-g_samples*e_samples))
    a_samples = a_samples[(np.abs(a_samples) < 5)]
    bin_edges = np.histogram_bin_edges(a_samples, bins='auto')
    print(bin_edges.size-1)
    hist, bin_edges = np.histogram(
                          a_samples, bins=int(bin_edges.size/8), density=True)
    a, p_a = (bin_edges[:-1]+bin_edges[+1:])/2, hist
    p_alphas[:,i] = np.interp(alphas, a, p_a, left=0, right=0)
    a_top = a[(p_a == p_a.max())][0]
    P, p, n = 1, 0, 0
    while (abs(P-0.95) >= 1e-8):
        n += 1
        if (P < 0.95):
            p -= p_a.max()/2**n
        if (P > 0.95):
            p += p_a.max()/2**n
        a_1 = brentq(dpap, -5, a_top, args=(p,))
        a_2 = brentq(dpap, a_top, +5, args=(p,))
        P = np.sum(np.all(((a_samples > a_1), (a_samples < a_2)), axis=0))/size
        if (n == 50): print('!'); break
    a_min[i], a_max[i] = a_1, a_2
    if (l == lamdas.max()):
        p = p_a[(a < -1/((1+(42.9*Mpc2m)/l)*np.exp(-(42.9*Mpc2m)/l)))].max()
        a_1 = brentq(dpap, -5, a_top, args=(p,))
        a_2 = brentq(dpap, a_top, +5, args=(p,))
        P = np.sum(np.all(((a_samples > a_1), (a_samples < a_2)), axis=0))/size
        print(P)
qcs = plt.contourf(lamdas, alphas, p_alphas, levels=1001)
plt.colorbar(qcs, label='$p(\\alpha)$')
plt.plot(lamdas, a_min, c='red')
plt.plot(lamdas, a_max, c='red')
plt.ylim((-1, +1))
plt.title('GW170817')
plt.xlabel('$\\lambda$(m)')
plt.ylabel('$\\alpha$')
plt.savefig('plot2.eps', dpi=800)
