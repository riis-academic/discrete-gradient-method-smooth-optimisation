#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:46:47 2018
"""


from __future__ import print_function, division
import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt
from scipy import linalg


np.random.seed(2)

n = int(5e2)  # problem dimension
A = np.random.normal(size=[n, n])
b = np.random.normal(size=n)

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

kappa = 10
eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])


Anew = linalg.sqrtm(AtA)
Atb = np.dot(Anew.T, b)


Lcol = np.diag(AtA)
L = eigvalNew[-1]
mu = eigvalNew[0]

x0 = np.zeros(n)

tau_c = 2 / Lcol
tau_r = 2 / Lcol
tau_mv = 2 / L
tau_grad = 1 / L

nIterOuter = 20000
nIterInner = 200
tolOuter = 1e-13
tolInner = 1e-13
Vtype = "linear_pd"  # linear_pd, linear
convex = True
printDG = 1


def Vfun(x):
    return np.linalg.norm(np.dot(Anew, x) - b) ** 2 / 2


def gradV(x):
    return np.dot(AtA, x) - Atb


def meanGradV(x, y):
    return np.dot(AtA, (x + y) / 2) - Atb


X, V, lastIter = dg.solve(
    n,
    x0,
    Vfun,
    tau_c,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    L,
    "itoh_abe_c",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG,
    Vtype,
    AtA,
    Atb,
)

X2, V2, lastIter2 = dg.solve(
    n,
    x0,
    Vfun,
    tau_r,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    L,
    "itoh_abe_r",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG,
    Vtype,
    AtA,
    Atb,
)

X3, V3, lastIter3 = dg.solve(
    n,
    x0,
    Vfun,
    tau_mv,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    L,
    "mean_value",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG,
    Vtype,
    AtA,
    Atb,
)


fig = plt.figure(figsize=(8, 6))
iters = list(range(lastIter + 1))
iters2 = list(range(lastIter2 + 1))
iters3 = list(range(lastIter3 + 1))
plt.semilogy(iters, V / V[0], linewidth=2, linestyle="-", marker="^", markersize=13, markevery=4)
plt.semilogy(iters2, V2 / V2[0], linewidth=2, linestyle="-", marker="o", markersize=13, markevery=9)
plt.semilogy(iters3, V3 / V3[0], linewidth=2, linestyle="-", marker="s", markersize=13, markevery=9)
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["CIA", "RIA", "Mean value/Gonzalez"], fontsize=20, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)

plt.show()
