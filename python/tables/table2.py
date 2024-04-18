#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:19:29 2018
"""
from scipy import linalg
import numpy as np
import discrete_gradient_compare as dg

np.random.seed(14)

n = int(5e2)  # problem dimension
A = np.random.normal(size=[n, n])
b = np.random.normal(size=n)

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

kappa = 100
eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])


Anew = linalg.sqrtm(AtA)
Atb = np.dot(Anew.T, b)

L = eigvalNew[-1]
mu = eigvalNew[0]

x0 = np.zeros(n)

tau_mv = 4 / L

nIterOuter = 50
nIterInner = 2000
tolOuter = 1e-6
tolInner = 1e-10
tolInner2 = tolInner
convex = True
printDG = 0


def Vfun(x):
    return np.linalg.norm(np.dot(Anew, x) - b) ** 2 / 2


def gradV(x):
    return np.dot(AtA, x) - Atb


def meanGradV(x, y):
    return np.dot(AtA, (x + y) / 2) - Atb


X, V, timeCost1, discrep1, lastIter = dg.solve(
    n,
    x0,
    Vfun,
    tau_mv,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    tolInner2,
    L,
    "mean_value",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG=printDG,
)

timeAvg1 = np.mean(timeCost1, axis=1)
discrepAvg1 = np.mean(discrep1, axis=1)

method_str = ["F", "R", "F+R", "fsolve"]

print_str = ""

print("Linear system")
for ii in range(len(method_str)):
    print_str = (
        method_str[ii] + " average time: " + str(timeAvg1[ii]) + ". Average error: " + str(discrepAvg1[ii]) + "."
    )
    print(print_str)

n = int(1e2)  # problem dimension
m = int(2e2)  # data variable dimension
np.random.seed(11)

y = np.sign(np.random.normal(size=m))
x = np.random.normal(size=[m, n])
C = 10

w0 = np.zeros(n)

L = 1 + C / 4 * np.sum((y * np.linalg.norm(x, axis=1)) ** 2)
Lvec = np.ones(n) + C / 4 * np.linalg.norm(y) ** 2 * np.linalg.norm(x, axis=0) ** 2

tau_mv = 4 / L

mu = 1
convex = True
printDG = 0


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


X, V, timeCost2, discrep2, lastIter = dg.solve(
    n,
    w0,
    Vfun,
    tau_mv,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    tolInner2,
    L,
    "mean_value",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG=printDG,
)


# %%

timeAvg2 = np.mean(timeCost2, axis=1)
discrepAvg2 = np.mean(discrep2, axis=1)

method_str = ["F", "R", "F+R", "fsolve"]

print_str = ""

print("Logistic regression")
for ii in range(len(method_str)):
    print_str = (
        method_str[ii] + " average time: " + str(timeAvg2[ii]) + ". Average error: " + str(discrepAvg2[ii]) + "."
    )
    print(print_str)


np.random.seed(11)

n = int(5e2)  # problem dimension

A = np.random.normal(size=[n, n])

AtA = np.dot(A.T, A)
AtAOld = np.array(AtA)

eigval, eigW = linalg.eigh(AtA, turbo=False)

c = eigW[:, 0]
kappa = 2000
eigvalNew = (eigval - eigval[0]) / (eigval[-1] - eigval[0]) * (kappa - 1) + 1

for ii in range(n):
    AtA += (eigvalNew[ii] - eigval[ii]) * np.outer(eigW[:, ii], eigW[:, ii])


Lcol = np.diag(AtA) + c * c
L = 2 * eigvalNew[-1] + 18
mu = -1
tau_mv = 4 / L

x0 = 100 * np.random.normal(size=n)

convex = False
printDG = 0


def Vfun(x):
    return np.dot(x, np.dot(AtA, x)) + 3 * np.sin(np.dot(c, x)) ** 2


def gradV(x):
    return 2 * np.dot(AtA, x) + 6 * np.cos(np.dot(c, x)) * np.sin(np.dot(c, x)) * c


def meanGradV(x, y):
    if all(x == y):
        return gradV(x)
    else:
        return (
            np.dot(AtA, (x + y) / 2)
            + 3 * (np.sin(np.dot(c, y)) ** 2 - np.sin(np.dot(c, x)) ** 2) / np.dot(c, y - x) * c
        )


X, V, timeCost3, discrep3, lastIter = dg.solve(
    n,
    x0,
    Vfun,
    tau_mv,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    tolInner2,
    L,
    "mean_value",
    mu,
    gradV,
    meanGradV,
    convex,
    printDG=printDG,
)


timeAvg3 = np.mean(timeCost3, axis=1)
discrepAvg3 = np.mean(discrep3, axis=1)

method_str = ["F", "R", "F+R", "fsolve"]

print_str = ""

print("Nonconvex problem")
for ii in range(len(method_str)):
    print_str = (
        method_str[ii] + " average time: " + str(timeAvg3[ii]) + ". Average error: " + str(discrepAvg3[ii]) + "."
    )
    print(print_str)
