#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:58:04 2018
"""

from scipy import linalg
import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt


np.random.seed(60)

n = int(5e2)  # problem dimension
A = np.random.uniform(size=[n, n])
b = np.random.normal(size=n)

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

kappa = 40
eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])


Anew = linalg.sqrtm(AtA)
Atb = np.dot(Anew.T, b)


Lcol = np.diag(AtA)
L = eigvalNew[-1]
mu = eigvalNew[0]

x0 = np.zeros(n)

tau_c1 = 1 / Lcol / np.sqrt(n)
tau_c2 = 2 / Lcol
tau_r = 2 / Lcol
tau_mv = 2 / L
tau_grad = 1 / L

nIterOuter = 1000
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
    tau_c1,
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

X4, V4, lastIter4 = dg.solve(
    n,
    x0,
    Vfun,
    tau_c2,
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
iters = list(range(lastIter + 1 - 5))
iters2 = list(range(lastIter2 + 1 - 5))
iters3 = list(range(lastIter3 + 1 - 5))
iters4 = list(range(lastIter4 + 1 - 5))
plt.semilogy(iters4, V4[4:-1] / V4[4], linewidth=2, linestyle="-", marker="^", markersize=13, markevery=130)
plt.semilogy(iters, V[4:-1] / V[4], linewidth=2, linestyle="-", marker="o", markersize=13, markevery=60)
plt.semilogy(iters2, V2[4:-1] / V2[4], linewidth=2, linestyle="-", marker="s", markersize=13, markevery=5)
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["CIA, heuristic", "CIA, proven", "RIA"], fontsize=20, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()
