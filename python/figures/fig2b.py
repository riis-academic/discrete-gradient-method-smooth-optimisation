#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:38:25 2018
"""

from scipy import linalg
import numpy as np
import discrete_gradient_solver as dg
import matplotlib.pyplot as plt

np.random.seed(101)


nIterOuter = 100
nIterInner = 200
tolOuter = 1e-8
tolInner = 1e-13
Vtype = "linear_pd"  # linear_pd, linear
convex = True
printDG = 1

n = int(5e2)  # problem dimension
kappa = 10


test = 100

V = np.empty(shape=[nIterOuter, test])
lastIter = np.empty(shape=test)

V.fill(np.nan)
lastIter.fill(np.nan)


A = np.random.normal(size=[n, n]) + 1
b = np.random.normal(size=n)

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])

Anew = linalg.sqrtm(AtA)
Atb = np.dot(Anew.T, b)

Lcol = np.diag(AtA)
L = eigvalNew[-1]
Lsum = np.linalg.norm(AtA)
mu = eigvalNew[0]

x0 = np.zeros(n)

tau_heur = 8 / Lcol
tau_prf = 2 / Lcol
beta_prf = 2 * n * np.max(Lcol)


def Vfun(x):
    return np.linalg.norm(np.dot(Anew, x) - b) ** 2 / 2


def gradV(x):
    return np.dot(AtA, x) - Atb


def meanGradV(x, y):
    return np.dot(AtA, (x + y) / 2) - Atb


for tt in range(test):

    print(tt)

    X, Vtmp, lastItertmp = dg.solve(
        n,
        x0,
        Vfun,
        tau_prf,
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

    V[0 : len(Vtmp), tt] = np.array(Vtmp)
    lastIter[tt] = np.array(lastItertmp)


lastIterMin = int(np.min(lastIter))
V = V[0:lastIterMin, :] / np.nanmax(V[:])


# %%


Vmean = np.exp(np.mean(np.log(V), axis=1))
# Vstdev = np.std(V,axis=1)
Vsort = np.array(V)
Vsort.sort(axis=1)


confidence = 10

indLow = int(test * confidence / 100 / 2)
indHigh = int(test * (1 - confidence / 100 / 2))

VLow = Vsort[:, indLow]
VHigh = Vsort[:, indHigh]


KK = 0

V2 = np.zeros(shape=lastIterMin)
V2[0] = 1
for ii in range(1, lastIterMin):
    V2[ii] = (1 - 2 * mu / beta_prf) ** n * V2[ii - 1]

fig = plt.figure(figsize=(8, 6))
iters = list(range(lastIterMin))
iters2 = list(range(lastIterMin))
plt.semilogy(Vmean, linewidth=2, linestyle="-", marker="^", markersize=13, markevery=5)
plt.fill_between(iters, VLow, VHigh, color="#d9e6f2", label="95% CI")

plt.semilogy(V2, linewidth=2, linestyle="--", color="k")
plt.ylabel("relative objective", fontsize=20)
plt.xlabel("iterates", fontsize=20)
plt.legend(["Observed rate", "Proven rate", "90% confidence"], fontsize=20, loc=0, frameon=False)
plt.tick_params(axis="both", labelsize=20)
plt.tick_params(axis="both", which="minor", labelsize=20)
plt.show()
