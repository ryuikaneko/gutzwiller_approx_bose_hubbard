#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import numpy as np
import scipy.linalg

def make_hamiltonian(z,nsps,t,U,mu,phi):
    H = np.zeros((nsps,nsps),dtype=np.float64)
    for n in range(nsps):
        H[n,n] = 0.5*U*n*(n-1) - mu*n + z*t*phi**2
    for n in range(nsps-1):
        H[n,n+1] = - z*t*phi*np.sqrt(n+1)
        H[n+1,n] = - z*t*phi*np.sqrt(n+1)
    return H

def calc_physquant(nsps,vec):
    norm2 = np.linalg.norm(vec)**2
    op_a = np.zeros((nsps,nsps),dtype=np.float64)
    op_n = np.zeros((nsps,nsps),dtype=np.float64)
    op_n2 = np.zeros((nsps,nsps),dtype=np.float64)
    for i in range(nsps-1):
        op_a[i,i+1] = np.sqrt(i+1)
    for i in range(nsps):
        op_n[i,i] = i
        op_n2[i,i] = i**2
    val_a = vec.dot(op_a.dot(vec))/norm2
    val_n = vec.dot(op_n.dot(vec))/norm2
    val_n2 = vec.dot(op_n2.dot(vec))/norm2
    return val_a, val_n, val_n2

def self_consistent_loop(z,nsps,t,U,mu):
    Nstep = 1000
    phi_old = nsps
    phi_new = nsps
    phi_eps = 1e-12
    dphi = 0.0
    for step in range(Nstep):
        H = make_hamiltonian(z,nsps,t,U,mu,phi_old)
#        print(H)
        ene, vec = scipy.linalg.eigh(H)
#        print(ene[0],vec[:,0])
        phi_new, n, n2 = calc_physquant(nsps,vec[:,0])
#        print(phi_new,n,n2)
        dphi = np.abs(phi_new - phi_old)
        phi_old = phi_new
#        print(step,phi_new,dphi)
        if dphi < phi_eps:
            break
    H = make_hamiltonian(z,nsps,t,U,mu,phi_new)
    ene, vec = scipy.linalg.eigh(H)
    phi, n, n2 = calc_physquant(nsps,vec[:,0])
    ene_t = -z*t*phi**2
    ene_U = 0.5*U*(n2-n)
    ene_mu = -mu*n
    return ene[0], ene_t, ene_U, ene_mu, vec[:,0], phi, dphi, n, n2

def main():
    z = 4
    U = 1.0
## nsps = n_{max states per site} = n_{max occupation} + 1
    nsps = 3
#    nsps = 10
    mu = 0.371
    list_t = np.linspace(0,0.1,101)
    print("# z nsps t U mu  ene ene_t ene_U ene_mu  phi error(phi) n n^2  vec")
    for t in list_t:
        ene, ene_t, ene_U, ene_mu, vec, phi, dphi, n, n2 = self_consistent_loop(z,nsps,t,U,mu)
        print(z,nsps,t,U,mu,end="  ")
        print(ene,ene_t,ene_U,ene_mu,end="  ")
        print(phi,dphi,n,n2,end="  ")
        print(' '.join(str(x) for x in vec))

if __name__ == "__main__":
    main()
