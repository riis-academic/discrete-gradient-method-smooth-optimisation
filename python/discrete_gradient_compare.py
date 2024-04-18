#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 14:04:15 2018
"""

import numpy as np
import scipy
import timeit


def solve(
    dim,
    x0,
    Vfun,
    tau,
    nIterOuter,
    nIterInner,
    tolOuter,
    tolInner,
    tolInner2,
    L,
    dg_type,
    mu=0,
    gradV=None,
    meanGradV=None,
    convex=True,
    printDG=0,
    Vtype=None,
    Vmat=None,
    Vdata=None,
):

    Vmat_new = None
    Vdata_new = None
    Vtype_new = None

    if convex == False:
        mu = -L

    X = np.empty((dim, nIterOuter + 1))
    V = np.empty((nIterOuter + 1))
    Tau = np.empty((dim, nIterOuter))
    timeCost = np.empty((4, nIterOuter))
    discrep = np.empty((4, nIterOuter))

    X.fill(np.nan)
    V.fill(np.nan)
    Tau.fill(np.nan)
    timeCost.fill(np.nan)
    discrep.fill(np.nan)

    X[:, 0] = np.array(x0)
    V[0] = Vfun(X[:, 0])

    for n in range(nIterOuter):
        if np.mod(n, 10) == 0:
            print(n)
        xx = np.array(X[:, n])

        X[:, n + 1], V[n + 1], timeCost[:, n], discrep[:, n] = discreteGradientInner(
            dim,
            xx,
            Vfun,
            V[n],
            tau,
            nIterInner,
            tolInner,
            tolInner2,
            L,
            mu,
            dg_type,
            printDG,
            gradV,
            meanGradV,
            Vtype_new,
            Vmat_new,
            Vdata_new,
            n,
        )

        if V[n] - V[n + 1] < tolOuter:
            if printDG >= 0:
                print("n: %3i, stop outer iter, diff: %3.2e < tol: %3.2e" % (n + 1, V[n] - V[n + 1], tolOuter))
            break

    lastIter = n + 1
    return X[:, 0 : lastIter + 1], V[0 : lastIter + 1], timeCost[:, 0:lastIter], discrep[:, 0:lastIter], lastIter


def discreteGradientInner(
    dim,
    xx,
    Vfun,
    Vxx,
    tau,
    nIterInner,
    tolInner,
    tolInner2,
    L,
    mu,
    dg_type,
    printDG,
    gradV,
    meanGradV,
    Vtype,
    Vmat,
    Vdata,
    n,
):

    yy = np.array(xx)
    timeCost = np.empty(4)
    discrep = np.empty(4)

    # %% General function

    if dg_type == "mean_value":
        Ldg = L / 2
        mudg = mu / 2

        def dgFun(y):
            return y - xx + tau * meanGradV(xx, y)

    elif dg_type == "gonzalez":
        Ldg = L / 2
        mudg = mu / 2

        def dgFun(y):
            gradY = gradV((xx + y) / 2)
            if np.all(y == xx):
                dgY = y - xx + tau * gradV(xx)
            else:
                dgY = (
                    y - xx + tau * (gradY + (Vfun(y) - Vxx - np.dot(gradY, y - xx)) / np.dot(y - xx, y - xx) * (y - xx))
                )
            return dgY

    sigma = 1
    yy = np.array(xx)
    dd = dgFun(yy)
    prevNorm = np.linalg.norm(dd)
    yy_prev = np.array(yy)
    yy = yy - sigma * dd
    dd = dgFun(yy)
    jj = 1
    tic = timeit.default_timer()
    relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)
    while any(abs(relerr) >= tolInner) and jj < nIterInner:
        prevNorm = np.linalg.norm(dd)
        yy_prev = np.array(yy)
        yy = yy - sigma * dd
        if np.max(np.abs(yy)) > 1e12:
            yy = yy + sigma * dd
        dd = dgFun(yy)
        relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)
        jj += 1
    timeCost[0] = timeit.default_timer() - tic
    discrep[0] = np.linalg.norm(dd)

    tic = timeit.default_timer()
    yy = scipy.optimize.fsolve(dgFun, xx, xtol=tolInner2, maxfev=nIterInner)
    dd = dgFun(yy)
    timeCost[3] = timeit.default_timer() - tic
    discrep[3] = np.linalg.norm(dd)
    sigma = 1
    jj = 1
    yy = np.array(xx)
    dd = dgFun(yy)
    yy_prev = np.array(yy)
    yy = yy - sigma * dd
    prevNorm = np.linalg.norm(dd)
    dd = dgFun(yy)
    tic = timeit.default_timer()
    relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)
    while any(abs(relerr) >= tolInner) and jj < nIterInner:
        yy_prev = np.array(yy)
        if mu >= 0:
            sigma2 = (1 + tau * mudg) / (1 + 2 * tau / 2 * mudg + Ldg**2 * tau**2)
        else:
            sigma2 = 1 / 2
        prevNorm = np.linalg.norm(dd)
        yy_tmp = yy - sigma * dd
        dd_tmp = dgFun(yy_tmp)
        jj += 1
        while np.linalg.norm(dd_tmp) >= prevNorm and jj < nIterInner:
            yy_tmp = yy - sigma2 * dd
            dd_tmp = dgFun(yy_tmp)
            jj += 1
            sigma2 = sigma2 / 2
        else:
            yy = yy_tmp
            dd = dd_tmp
        relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)

    timeCost[2] = timeit.default_timer() - tic
    discrep[2] = np.linalg.norm(dd)

    yy = np.array(xx)
    if mu >= 0:
        sigma = (1 + tau * mudg) / (1 + 2 * tau / 2 * mudg + Ldg**2 * tau**2)
    else:
        sigma = 1 / 2

    dd = dgFun(yy)
    yy_prev = np.array(yy)
    yy = yy - sigma * dd
    dd = dgFun(yy)
    jj = 1
    tic = timeit.default_timer()
    relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)
    while any(abs(relerr) >= tolInner) and jj < nIterInner:
        yy_prev = np.array(yy)
        prevNorm = np.linalg.norm(dd)
        yy = yy - sigma * dd
        dd = dgFun(yy)
        jj += 1
        relerr = (yy - yy_prev) / np.where(yy_prev == 0, 1, yy_prev)
    timeCost[1] = timeit.default_timer() - tic
    discrep[1] = np.linalg.norm(dd)

    return yy, Vfun(yy), timeCost, discrep
