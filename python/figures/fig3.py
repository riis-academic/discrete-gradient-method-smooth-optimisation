#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:32:20 2018
"""


from scipy import linalg
import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt


np.random.seed(7)

n = int(8e2)  # problem dimension
m = int(4e2)
A = np.random.normal(size=[m, n])
b = np.random.normal(size=m)

AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)

eigval, eigW = linalg.eigh(AtA, turbo=False)
eigval = eigval * (np.abs(eigval) >= 1e-14)

k = 0
while eigval[k] == 0:
    k += 1


Lcol = np.diag(AtA)
L = eigval[-1]
mu = 0

x0 = np.zeros(n)

tau_c = 2 / Lcol
tau_r = 2 / Lcol
tau_rs = 2 / L
tau_mv = 2 / L
tau_grad = 1 / L

nIterOuter = 1000
nIterInner = 200
tolOuter = 1e-10
tolInner = 1e-16
Vtype = "linear_pd"  # linear_pd, linear
convex = True
printDG = 1


def Vfun(x):
    return np.linalg.norm(np.dot(A, x) - b) ** 2 / 2


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
plt.semilogy(iters, V / V[0], linewidth=2, linestyle="-", marker="^", markersize=13, markevery=7)
plt.semilogy(iters2, V2 / V2[0], linewidth=2, linestyle="-", marker="o", markersize=13, markevery=10)
plt.semilogy(iters3, V3 / V3[0], linewidth=2, linestyle="-", marker="s", markersize=13, markevery=18)
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["CIA", "RIA", "Mean value/Gonzalez"], fontsize=20, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()
