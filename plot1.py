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
print(g[(p_g == p_g.max())][0])

def P_normal(p):
    P = 1
    for i in range(0, g.size, 1):
        if (p_g[i+1] > p):
            g_1 = np.interp(p, p_g[i:i+2:+1], g[i:i+2:+1])
            P -= ((g[1:i+1]-g[:i])*(p_g[:i]+p_g[1:i+1])).sum()/2
            P -= (g_1-g[i])*(p_g[i]+p)/2
            break
    for i in range(g.size-1, -1, -1):
        if (p_g[i-1] > p):
            g_2 = np.interp(p, p_g[i:i-2:-1], g[i:i-2:-1])
            P -= ((g[i+1:]-g[i:-1])*(p_g[i+1:]+p_g[i:-1])).sum()/2
            P -= (g[i]-g_2)*(p_g[i]+p)/2
            break
    return P, g_1, g_2

P, p, n = 1, 0, 0
while (abs(P-0.6826) >= 0.0001):
    n += 1
    if (P < 0.6826):
        p -= p_g.max()/2**n
    if (P > 0.6826):
        p += p_g.max()/2**n
    P, g_1, g_2 = P_normal(p)
print(P, g_1, g_2)

P, p, n = 1, 0, 0
while (abs(P-0.95) >= 0.0001):
    n += 1
    if (P < 0.95):
        p -= p_g.max()/2**n
    if (P > 0.95):
        p += p_g.max()/2**n
    P, g_1, g_2 = P_normal(p)
print(P, g_1, g_2)

for i in range(g.size-1, -1, -1):
    if (g[i-1] < 1):
        p = np.interp(1, g[i-1:i+1], p_g[i-1:i+1])
        break
for i in range(0, g.size, 1):
    if (p_g[i+1] > p):
        g_3 = np.interp(p, p_g[i:i+2], g[i:i+2])
        break

plt.plot(g, p_g)
plt.axvline(g_1, c='tab:grey', ls='--')
plt.axvline(g_2, c='tab:grey', ls='--')
plt.axvline(1, c='tab:grey', ls=':')
plt.plot([g_3, 1], [p, p], c='tab:grey', ls=':')
plt.axvline(g_3, c='tab:grey', ls=':')
plt.fill_between(np.linspace(g_3, 1, 1001),
                 np.interp(np.linspace(g_3, 1, 1001), g, p_g),
                 color='lightblue')
plt.title('GW170817')
plt.xlim((-0.1, +2.1))
plt.ylim((-0.10, +1.35))
plt.xlabel('$G_s/G$')
plt.ylabel('$p(G_s/G)$')
plt.grid()
plt.savefig('plot1.eps', dpi=800)

plt.clf()

from astropy.constants import c as light_speed
light_speed = light_speed.to('Mpc/yr').value

f = h5py.File('GW170817_GWTC-1.hdf5', 'r')
dset = f['IMRPhenomPv2NRT_lowSpin_posterior']
d_l = dset['luminosity_distance_Mpc']
size = d_l.size**2
sample = ((1-
         (np.random.choice(d_l, size=size)
         /np.random.normal(loc=42.9, scale=3.2, size=size))**2)
         /np.random.normal(loc=42.9, scale=3.2, size=size)*light_speed)
hist, bin_edges = np.histogram(sample, bins='auto', density=True)
print(bin_edges.size-1)
hist, bin_edges = np.histogram(sample, bins=int(bin_edges.size/8), density=True)
g, p_g = (bin_edges[:-1]+bin_edges[+1:])/2, hist
print(g[(p_g == p_g.max())][0])

def P_normal(p):
    P = 1
    for i in range(0, g.size, 1):
        if (p_g[i+1] > p):
            g_1 = np.interp(p, p_g[i:i+2:+1], g[i:i+2:+1])
            P -= ((g[1:i+1]-g[:i])*(p_g[:i]+p_g[1:i+1])).sum()/2
            P -= (g_1-g[i])*(p_g[i]+p)/2
            break
    for i in range(g.size-1, -1, -1):
        if (p_g[i-1] > p):
            g_2 = np.interp(p, p_g[i:i-2:-1], g[i:i-2:-1])
            P -= ((g[i+1:]-g[i:-1])*(p_g[i+1:]+p_g[i:-1])).sum()/2
            P -= (g[i]-g_2)*(p_g[i]+p)/2
            break
    return P, g_1, g_2

P, p, n = 1, 0, 0
while (abs(P-0.6826) >= 0.0001):
    n += 1
    if (P < 0.6826):
        p -= p_g.max()/2**n
    if (P > 0.6826):
        p += p_g.max()/2**n
    P, g_1, g_2 = P_normal(p)
print(P, g_1, g_2)

P, p, n = 1, 0, 0
while (abs(P-0.95) >= 0.0001):
    n += 1
    if (P < 0.95):
        p -= p_g.max()/2**n
    if (P > 0.95):
        p += p_g.max()/2**n
    P, g_1, g_2 = P_normal(p)
print(P, g_1, g_2)

plt.plot(g, p_g)
plt.axvline(g_1, c='tab:grey', ls='--')
plt.axvline(g_2, c='tab:grey', ls='--')
plt.title('GW170817')
plt.xlim((-0.75e-8, +0.75e-8))
plt.ylim((-0.10e+8, +1.85e+8))
plt.xlabel('$\\dot{G}/G$')
plt.ylabel('$p(\\dot{G}/G)$')
plt.grid()
plt.savefig('plot1.1.eps', dpi=800)
