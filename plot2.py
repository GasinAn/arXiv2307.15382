import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.subplots(layout='constrained')

f = h5py.File('GW170817_GWTC-1.hdf5', 'r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = dset['luminosity_distance_Mpc']
size = d_l.size**2
sample = (np.random.choice(d_l, size=size)
         /np.random.normal(loc=42.9, scale=3.2, size=size))**2
hist, bin_edges = np.histogram(sample, bins='auto', density=True)
print(bin_edges.size-1)
hist, bin_edges = np.histogram(sample, bins=int(bin_edges.size/8), density=True)
g, p_g = (bin_edges[:-1]+bin_edges[+1:])/2, hist
g_ = np.linspace(g.min(), g.max(), 1001)
g, p_g = g_, np.interp(g_, g, p_g)

lamdas, alphas = np.linspace(0, 3, 1001), np.linspace(-8, +8, 1001)
p_alphas = np.empty((alphas.size, lamdas.size))
for i in range(lamdas.size):
    lamda = lamdas[i]
    if (lamda == 0):
        e = np.array(0)
    else:
        e = np.array((1+1/lamda)*np.exp(-1/lamda))
    alpha = -((1-g)/(1-g*e))
    p_alpha = p_g*((1-e)/(1+alpha*e)**2)
    alpha_i = np.sort_complex(alpha+p_alpha*1j)
    p_alphas[:,i] = np.interp(alphas, alpha_i.real, alpha_i.imag)
qcs = plt.contourf(lamdas, alphas, p_alphas, levels=1001)
plt.colorbar(qcs, label='$p(\\alpha)$')

f = h5py.File('GW170817_GWTC-1.hdf5', 'r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = dset['luminosity_distance_Mpc']
size = d_l.size**2
sample = (np.random.choice(d_l, size=size)
         /np.random.normal(loc=42.9, scale=3.2, size=size))**2
hist, bin_edges = np.histogram(sample, bins='auto', density=True)
print(bin_edges.size-1)
hist, bin_edges = np.histogram(sample, bins=int(bin_edges.size/8), density=True)
g, p_g = (bin_edges[:-1]+bin_edges[+1:])/2, hist

from scipy.optimize import brentq
def dpap(a, e, p):
    g_ = (1+a)/(1+a*e)
    p_g_ = np.interp(g_, g, p_g, left=0, right=0)
    return p_g_*((1-e)/(1+a*e)**2)-p

ls = np.sort(np.hstack((np.linspace(0, 3, 101), np.linspace(0.9, 1.0, 101))))
as_1_1, as_1_2, as_2_1, as_2_2 = [], [], [], []
for i in range(ls.size):
    print(i)
    l = ls[i]
    if (l == 0):
        e = np.float64(0)
    else:
        e = np.float64((1+1/l)*np.exp(-1/l))
    a = -((1-sample)/(1-sample*e))
    a = a[(abs(a) < 10)]
    a_ = np.linspace(-10, +10, 10001)
    p_a_ = dpap(a_, e, 0)
    if (l < 0.8):
        a_max_2 = a_[(p_a_ == p_a_.max())][0]
        P, p, n = 1, 0, 0
        while (abs(P-0.95) >= 0.0001):
            n += 1
            if (P < 0.95):
                p -= p_a_.max()/2**n
            if (P > 0.95):
                p += p_a_.max()/2**n
            a_2_1 = brentq(dpap, np.max((-10, -1/e)), a_max_2, args=(e, p))
            a_2_2 = brentq(dpap, a_max_2, +10, args=(e, p))
            P = np.sum(np.all(((a > a_2_1), (a < a_2_2)), axis=0))/size
            if (n == 50): print('!'); break
        as_2_1.append(a_2_1), as_2_2.append(a_2_2)
        i_ = i
    else:
        p_a_1_max = p_a_[(a_ < -1/e)].max()
        a_max_2 = a_[(p_a_ == p_a_.max())][0]
        a_2_1 = brentq(dpap, -1/e, a_max_2, args=(e, p_a_1_max))
        a_2_2 = brentq(dpap, a_max_2, +10, args=(e, p_a_1_max))
        P = np.sum(np.all(((a > a_2_1), (a < a_2_2)), axis=0))/size
        if (P > 0.95):
            P, p, n = 1, 0, 0
            while (abs(P-0.95) >= 0.0001):
                n += 1
                if (P < 0.95):
                    p -= p_a_.max()/2**n
                if (P > 0.95):
                    p += p_a_.max()/2**n
                a_2_1 = brentq(dpap, -1/e, a_max_2, args=(e, p))
                a_2_2 = brentq(dpap, a_max_2, +10, args=(e, p))
                P = np.sum(np.all(((a > a_2_1), (a < a_2_2)), axis=0))/size
                if (n == 50): print('!'); break
            as_2_1.append(a_2_1), as_2_2.append(a_2_2)
            i_ = i
        else:
            a_max_1 = a_[(p_a_ == p_a_1_max)][0]
            P, p, n = 1, 0, 0
            while (abs(P-0.95) >= 0.0001):
                n += 1
                if (P < 0.95):
                    p -= p_a_1_max/2**n
                if (P > 0.95):
                    p += p_a_1_max/2**n
                a_1_1 = brentq(dpap, -10, a_max_1, args=(e, p))
                a_1_2 = brentq(dpap, a_max_1, -1/e, args=(e, p))
                a_2_1 = brentq(dpap, -1/e, a_max_2, args=(e, p))
                a_2_2 = brentq(dpap, a_max_2, +10, args=(e, p))
                P_1 = np.sum(np.all(((a > a_1_1), (a < a_1_2)), axis=0))/size
                P_2 = np.sum(np.all(((a > a_2_1), (a < a_2_2)), axis=0))/size
                P = P_1+P_2
                if (n == 50): print('!'); break
            as_1_1.append(a_1_1), as_1_2.append(a_1_2)
            as_2_1.append(a_2_1), as_2_2.append(a_2_2)

plt.plot(ls[i_+1:], np.array(as_1_1), c='red')
plt.plot(ls[i_+1:], np.array(as_1_2), c='red')
plt.plot(ls, np.array(as_2_1), c='red')
plt.plot(ls, np.array(as_2_2), c='red')

plt.title('GW170817')
plt.xlabel('$\\lambda/D_L$')
plt.ylabel('$\\alpha$')
plt.show()
