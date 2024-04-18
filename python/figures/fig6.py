#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:49:19 2018
"""

from scipy import linalg
import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt


np.random.seed(1000)

n = int(50)  # problem dimension
A = np.random.normal(size=[n, n])

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

kappa = 50
eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])

c = eigW[:, 0]


Lcol = np.diag(AtA) + c * c
L = 2 * eigvalNew[-1] + 18
mu = eigvalNew[0] / eigvalNew[-1] / 32

x0 = 100 * np.random.normal(size=n)

tau_c = 2 / Lcol
tau_r = 2 / Lcol
tau_rs = 2 / L
tau_mv = 2 / L
tau_gon = 2 / L

nIterOuter = 1000
nIterInner = 200
tolOuter = 1e-12
tolInner = 1e-12
convex = False
printDG = 1


def Vfun(x):
    return np.dot(x, np.dot(AtA, x)) + 3 * np.sin(np.dot(c, x)) ** 2


def gradV(x):
    return 2 * np.dot(AtA, x) + 6 * np.cos(np.dot(c, x)) * np.sin(np.dot(c, x)) * c


def meanGradV(x, y):
    if all(x == y):
        return gradV(x)
    else:
        return (
            2 * np.dot(AtA, (x + y) / 2)
            + 3 * (np.sin(np.dot(c, y)) ** 2 - np.sin(np.dot(c, x)) ** 2) / np.dot(c, y - x) * c
        )


X4, V4, lastIter4 = dg.solve(
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
)

X5, V5, lastIter5 = dg.solve(
    n,
    x0,
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
)


gradV1 = np.zeros(len(V))
for i in range(len(V) - 1):
    gradV1[i] = np.linalg.norm(gradV(X[:, i]))
gradV2 = np.zeros(len(V2))
for i in range(len(V2) - 1):
    gradV2[i] = np.linalg.norm(gradV(X2[:, i]))
gradV4 = np.zeros(len(V4))
for i in range(len(V4) - 1):
    gradV4[i] = np.linalg.norm(gradV(X4[:, i]))
gradV5 = np.zeros(len(V5))
for i in range(len(V5) - 1):
    gradV5[i] = np.linalg.norm(gradV(X5[:, i]))


fig = plt.figure(figsize=(8, 6))
iters = list(range(lastIter + 1))
iters2 = list(range(lastIter2 + 1))
iters4 = list(range(lastIter4 + 1))
iters5 = list(range(lastIter5 + 1))
plt.semilogy(iters, V / V[0], linewidth=2, linestyle="-", marker="^", markersize=13, markevery=20)
plt.semilogy(iters2, V2 / V[0], linewidth=2, linestyle="-", marker="o", markersize=13, markevery=30)
plt.semilogy(iters4, V4 / V[0], linewidth=2, linestyle="-", marker="s", markersize=13, markevery=80)
plt.semilogy(iters5, V5 / V[0], linewidth=3, linestyle=":", marker="*", markersize=13, markevery=50)
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["CIA", "RIA", "Mean value", "Gonzalez"], fontsize=20, loc=0, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()

fig = plt.figure(figsize=(8, 6))
iters = list(range(lastIter + 1))
iters2 = list(range(lastIter2 + 1))
iters4 = list(range(lastIter4 + 1))
iters5 = list(range(lastIter5 + 1))
plt.semilogy(iters, gradV1 / gradV1[0], linewidth=2, linestyle="-", marker="^", markersize=13, markevery=20)
plt.semilogy(iters2, gradV2 / gradV1[0], linewidth=2, linestyle="-", marker="o", markersize=13, markevery=30)
plt.semilogy(iters4, gradV4 / gradV1[0], linewidth=2, linestyle="-", marker="s", markersize=13, markevery=80)
plt.semilogy(iters5, gradV5 / gradV1[0], linewidth=3, linestyle=":", marker="*", markersize=13, markevery=50)
plt.ylabel("norm of gradient (normalised)", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["CIA", "RIA", "Mean value", "Gonzalez"], fontsize=20, loc=0, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()
