#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:43:32 2018
"""

import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt


n = int(1e2)  # problem dimension
m = int(2e2)  # data variable dimension
np.random.seed(10)

y = np.sign(np.random.normal(size=m))
x = np.random.normal(size=[m, n])
C = 1 / 2


w0 = np.zeros(n)

L = 1 + C / 4 * np.sum((y * np.linalg.norm(x, axis=1)) ** 2)
Lvec = np.ones(n) + C / 4 * np.linalg.norm(y) ** 2 * np.linalg.norm(x, axis=0) ** 2
mu = 1
tau_c = 2 / Lvec
tau_r = 2 / Lvec
# tau_rs = 1
tau_mv = 2 / L
tau_gon = 2 / L
tau_grad = 1 / L

nIterOuter = 400  # 20000
nIterInner = 100
tolOuter = 1e-13
tolInner = 1e-10
mu = 1
convex = True
printDG = 1


def Vfun(w):
    return C * np.sum(np.log(np.ones(m) + np.exp(-y * np.dot(x, w)))) + 1 / 2 * np.linalg.norm(w) ** 2


def gradV(w):
    return w + C * np.sum(-y * x.T * np.exp(-y * np.dot(x, w)) / (1 + np.exp(-y * np.dot(x, w))), axis=1)


def meanGradV(w, z):
    if all(w == z):
        DGrad = gradV(w)
    else:
        DGrad = (w + z) / 2 + C * np.sum(
            x.T / np.dot(x, w - z) * np.log((1 + np.exp(-y * np.dot(x, w))) / (1 + np.exp(-y * np.dot(x, z)))), axis=1
        )
    return DGrad


X, V, lastIter = dg.solve(
    n,
    w0,
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
)

X2, V2, lastIter2 = dg.solve(
    n,
    w0,
    Vfun,
    tau_gon,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    L,
    "gonzalez",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG,
)

X3, V3, lastIter3 = dg.solve(
    n,
    w0,
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
)

X4, V4, lastIter4 = dg.solve(
    n,
    w0,
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
)


fig = plt.figure(figsize=(8, 6))
iters = list(range(lastIter + 1))
iters2 = list(range(lastIter2 + 1))
iters3 = list(range(lastIter3 + 1))
iters4 = list(range(lastIter4 + 1))
plt.semilogy(
    iters3,
    (V3 - 42.6183912085) / (V3[0] - 42.61839120859),
    linewidth=2,
    linestyle="-",
    marker="s",
    markersize=13,
    markevery=58,
)
plt.semilogy(
    iters4,
    (V4 - 42.6183912085) / (V4[0] - 42.61839120859),
    linewidth=3,
    linestyle=":",
    marker="*",
    markersize=13,
    markevery=40,
)
plt.semilogy(
    iters,
    (V - 42.6183912085) / (V[0] - 42.61839120859),
    linewidth=2,
    linestyle="-",
    marker="^",
    markersize=13,
    markevery=40,
)
plt.semilogy(
    iters2,
    (V2 - 42.6183912085) / (V2[0] - 42.61839120859),
    linewidth=3,
    linestyle=":",
    marker="o",
    markersize=13,
    markevery=58,
)
plt.legend(["CIA", "RIA", "Mean value", "Gonzalez"], fontsize=20, frameon=False)
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()
