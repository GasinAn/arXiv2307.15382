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

def P_normal_1(p):
    P = 0
    for i in range(0, a_1.size, 1):
        if (p_a_1[i+1] > p):
            i_1 = i+1
            a_1_1 = np.interp(p, p_a_1[i:i+2:+1], a_1[i:i+2:+1])
            break
    for i in range(a_1.size-1, -1, -1):
        if (p_a_1[i-1] > p):
            i_2 = i-1
            a_1_2 = np.interp(p, p_a_1[i:i-2:-1], a_1[i:i-2:-1])
            break
    P += ((a_1[i_1+1:i_2+1]-a_1[i_1:i_2])
         *(p_a_1[i_1:i_2]+p_a_1[i_1+1:i_2+1])).sum()/2
    P += (a_1[i_1]-a_1_1)*(p+p_a_1[i_1])/2
    P += (a_1_2-a_1[i_2])*(p+p_a_1[i_2])/2
    return P, a_1_1, a_1_2

def P_normal_2(p):
    P = 0
    for i in range(0, a_2.size, 1):
        if (p_a_2[i+1] > p):
            i_1 = i+1
            a_2_1 = np.interp(p, p_a_2[i:i+2:+1], a_2[i:i+2:+1])
            break
    for i in range(a_2.size-1, -1, -1):
        if (p_a_2[i-1] > p):
            i_2 = i-1
            a_2_2 = np.interp(p, p_a_2[i:i-2:-1], a_2[i:i-2:-1])
            break
    P += ((a_2[i_1+1:i_2+1]-a_2[i_1:i_2])
         *(p_a_2[i_1:i_2]+p_a_2[i_1+1:i_2+1])).sum()/2
    P += (a_2[i_1]-a_2_1)*(p+p_a_2[i_1])/2
    P += (a_2_2-a_2[i_2])*(p+p_a_2[i_2])/2
    return P, a_2_1, a_2_2

g = (np.random.choice(d_l, size=size)
    /np.random.normal(loc=42.9, scale=3.2, size=size))**2
ls = np.linspace(0, 3, 101)
as_1_1, as_1_2, as_2_1, as_2_2 = [], [], [], []
for i in range(ls.size):
    print(i)
    l = ls[i]
    if (l == 0):
        e = np.array(0)
    else:
        e = np.array((1+1/l)*np.exp(-1/l))
    a = -((1-g)/(1-g*e))
    a = a[np.where(abs(a) < 20)]
    rate = a.size/size
    hist, bin_edges = np.histogram(a, bins='auto', density=True)
    a, p_a = (bin_edges[:-1]+bin_edges[+1:])/2, hist
    where_1, where_2 = np.where(a < -1/e), np.where(a > -1/e)
    a_1, a_2 = a[where_1], a[where_2]
    p_a_1, p_a_2 = p_a[where_1], p_a[where_2]
    if (a_1.size == 0):
        P, p, n = 0, p_a_2.max(), 0
        while (abs(P-0.95) >= 0.0001):
            n += 1
            if (P < 0.95):
                p -= p_a_2.max()/2**n
            if (P > 0.95):
                p += p_a_2.max()/2**n
            P, a_2_1, a_2_2 = P_normal_2(p)
            P *= rate
            if (n == 50):
                print(i, '!')
                break
        as_2_1.append(a_2_1), as_2_2.append(a_2_2)
        i_ = i
    else:
        P, a_2_1, a_2_2 = P_normal_2(p_a_1.max())
        P *= rate
        if (P > 0.95):
            P, p, n = 0, p_a_2.max(), 0
            while (abs(P-0.95) >= 0.0001):
                n += 1
                if (P < 0.95):
                    p -= p_a_2.max()/2**n
                if (P > 0.95):
                    p += p_a_2.max()/2**n
                P, a_2_1, a_2_2 = P_normal_2(p)
                P *= rate
                if (n == 50):
                    print(i, '!')
                    break
            as_2_1.append(a_2_1), as_2_2.append(a_2_2)
            i_ = i
        else:
            P, p, n = 0, p_a_1.max(), 0
            while (abs(P-0.95) >= 0.0001):
                n += 1
                if (P < 0.95):
                    p -= p_a_1.max()/2**n
                if (P > 0.95):
                    p += p_a_1.max()/2**n
                P_1, a_1_1, a_1_2 = P_normal_1(p)
                P_2, a_2_1, a_2_2 = P_normal_2(p)
                P = P_1+P_2
                P *= rate
                if (n == 50):
                    print(i, '!')
                    break
            print(a_1_1, a_1_2, a_2_1, a_2_2)
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
