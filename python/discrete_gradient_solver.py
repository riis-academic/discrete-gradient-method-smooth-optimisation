#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:28:10 2018
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
    Vfun_IA=None,
    gradV_IA=None,
    rootFun=None,
):

    TIC = timeit.default_timer()

    if Vtype == "linear":
        Vmat_new = np.dot(Vmat.T, Vmat)
        Vdata_new = np.dot(Vmat.T, Vdata)
        Vtype_new = "linear_pd"
    elif Vtype == "linear_pd":
        Vmat_new = np.array(Vmat)
        Vdata_new = np.array(Vdata)
        Vtype_new = "linear_pd"
    else:
        Vmat_new = None
        Vdata_new = None
        Vtype_new = None

    if convex == False:
        mu = -L

    if dg_type == "gonzalez":
        dgTypePrint = "Gonzalez"
    elif dg_type == "mean_value":
        dgTypePrint = "Mean value"
    elif dg_type == "itoh_abe_c":
        dgTypePrint = "Itoh--Abe cyclic"
    elif dg_type == "itoh_abe_r":
        dgTypePrint = "Itoh--Abe random coordinate"
    elif dg_type == "itoh_abe_rs":
        dgTypePrint = "Itoh--Abe random search"
    elif dg_type == "itoh_abe_rc":
        dgTypePrint = "Itoh--Abe random cyclic"
    elif dg_type == "gradient":
        dgTypePrint = "Gradient"

    X = np.empty((dim, nIterOuter + 1))
    V = np.empty((nIterOuter + 1))
    Tau = np.empty((dim, nIterOuter))

    X.fill(np.nan)
    V.fill(np.nan)
    Tau.fill(np.nan)

    X[:, 0] = np.array(x0)
    V[0] = Vfun(X[:, 0])

    tic = timeit.default_timer()

    for n in range(nIterOuter):
        if printDG >= 1:
            print("n: %3i, V: %3.7e, time: %3.2es. DG: %s" % (n + 1, V[n], timeit.default_timer() - tic, dgTypePrint))

        if dg_type == "gradient":
            X[:, n + 1] = X[:, n] - tau * gradV(X[:, n])
            V[n + 1] = Vfun(X[:, n + 1])
        else:

            xx = np.array(X[:, n])

            X[:, n + 1], V[n + 1] = discreteGradientInner(
                dim,
                xx,
                Vfun,
                V[n],
                tau,
                nIterInner,
                tolInner,
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
                Vfun_IA,
                gradV_IA,
                rootFun,
            )

        if V[n] - V[n + 1] < tolOuter:
            if printDG >= 0:
                print("n: %3i, stop outer iter, diff: %3.2e < tol: %3.2e" % (n + 1, V[n] - V[n + 1], tolOuter))
            break

    if printDG >= 0 and n is nIterOuter - 1:
        print("n: %3i, stop outer iter because nIter reached" % (n + 1))

    lastIter = n + 1

    print(timeit.default_timer() - TIC)
    return X[:, 0 : lastIter + 1], V[0 : lastIter + 1], lastIter


def discreteGradientInner(
    dim,
    xx,
    Vfun,
    Vxx,
    tau,
    nIterInner,
    tolInner,
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
    Vfun_IA,
    gradV_IA,
    rootFun,
):

    yy = np.array(xx)
    Vyy = Vxx
    # %% General function

    if Vtype == None:
        if dg_type in ["itoh_abe_r", "itoh_abe_c", "itoh_abe_rc"]:
            if dg_type == "itoh_abe_rc":
                jj_vec = np.random.choice(range(dim), dim, replace=False)
            for ii in range(dim):
                if dg_type == "itoh_abe_r":
                    jj = np.random.choice(range(dim))
                elif dg_type == "itoh_abe_rc":
                    jj = jj_vec[ii]
                else:
                    jj = ii
                if type(tau) in [float, np.float64]:
                    Tau = tau
                else:
                    Tau = tau[jj]

                if Vfun_IA is None and rootFun is None:

                    def iaFun(y):
                        if y == yy[jj]:
                            iaF = yy[jj] - Tau * gradV(yy)[jj]
                        else:
                            zz = np.array(yy)
                            zz[jj] = y
                            iaF = yy[jj] - Tau * (Vfun(zz) - Vyy) / (y - yy[jj])
                        return iaF

                    try:
                        y_new = scipy.optimize.fixed_point(iaFun, yy[jj] + 1e-5, maxiter=40)
                    except RuntimeError:
                        y_new = dg_fixed_point(yy, Vfun, Vyy, gradV, Tau, "coordinate", coord=jj) + yy[jj]
                elif rootFun is None:
                    if np.mod(jj, 1000) == 0:
                        print(jj / dim)
                    Vyy = Vfun_IA(yy, jj)

                    def iaFun(y):
                        if y == yy[jj]:
                            iaF = yy[jj] - Tau * gradV_IA(yy, jj)
                        else:
                            zz = np.array(yy)
                            zz[jj] = y
                            iaF = yy[jj] - Tau * (Vfun_IA(zz, jj) - Vyy) / (y - yy[jj])
                        return iaF

                    y_new = dg_fixed_point_IA(yy, Vfun_IA, Vfun_IA(yy, ii), gradV_IA, Tau, coord=jj) + yy[jj]
                else:
                    if np.mod(jj, 1000) == 0:
                        print(jj / dim)
                    y_new = rootFun(yy, jj, Tau)
                Vyy = Vyy - (yy[jj] - y_new) ** 2 / Tau
                yy[jj] = y_new
        elif dg_type == "itoh_abe_rs":
            for ii in range(dim):
                f = np.random.normal(size=dim)
                f = f / np.linalg.norm(f)

                Vyy = Vfun(yy)

                def iaFun(aa):
                    if aa == 0:
                        iaF = -tau * np.dot(gradV(yy), f)
                    else:
                        iaF = -tau * (Vfun(yy - aa * f) - Vyy) / aa
                    return iaF

                try:
                    a_new = scipy.optimize.fixed_point(iaFun, 1e-5, maxiter=40)
                except RuntimeError:
                    a_new = -dg_fixed_point(yy, Vfun, Vyy, gradV, tau, "search", dd=f)
                yy = yy - a_new * f
        else:
            if dg_type == "mean_value":
                Ldg = L / 2
                mu_dg = mu / 2

                def dgFun(y):
                    return y - xx + tau * meanGradV(xx, y)

            elif dg_type == "gonzalez":
                Ldg = L / 2
                mu_dg = mu / 2

                def dgFun(y):
                    gradY = gradV((xx + y) / 2)
                    if np.all(y == xx):
                        dgY = y - xx + tau * gradV(xx)
                    else:
                        dgY = (
                            y
                            - xx
                            + tau
                            * (gradY + (Vfun(y) - Vxx - np.dot(gradY, y - xx)) / np.dot(y - xx, y - xx) * (y - xx))
                        )
                    return dgY

            sigma = 1
            dd = dgFun(yy)
            dd_prev = np.array(dd)
            prevNorm = np.linalg.norm(dd)
            yy_prev = np.array(yy)
            yy = yy - sigma * dd
            dd = dgFun(yy)
            jj = 1
            kk = 1
            while np.linalg.norm(dd) >= tolInner and jj < nIterInner:
                if np.linalg.norm(dd) < prevNorm:
                    sigma = 1
                    prevNorm = np.linalg.norm(dd)
                    yy_prev = np.array(yy)
                    dd_prev = np.array(dd)
                    yy = yy - sigma * dd
                    dd = dgFun(yy)
                elif mu_dg >= 0 and Ldg > 0:
                    sigma = float(sigma) * (1 + tau * mu_dg) / (1 + tau**2 * Ldg**2 + 2 * tau * mu_dg)
                    yy = yy_prev - sigma * dd_prev
                    dd = dgFun(yy)
                    kk += 1
                else:
                    sigma = float(sigma) / 2
                    yy = yy_prev - sigma * dd_prev
                    dd = dgFun(yy)
                    kk += 1
                if printDG >= 2:
                    print("k: %3i, ||d||: %3.7e" % (jj + 1, np.linalg.norm(dd)))
                jj += 1

            if printDG >= 1 and jj == nIterInner:
                print("k: %3i, stop inner iter because nIter reached, d: %3.2e." % (jj + 1, np.linalg.norm(dd)))
            else:
                print("k: %3i, stop inner iter, diff: %3.2e < tol: %3.2e." % (jj + 1, np.linalg.norm(dd), tolInner))

    #
    #
    #            if mu >= 0 or (mu < 0 and tau < 1/Ldg):
    #                sigma = (1+tau*mu)/(1+2*tau*mu+Ldg**2*tau**2)
    #            elif tau == 1/Ldg:
    #                sigma = (1+tau*mu)/(1+2*tau*mu+Ldg**2*tau**2)/4
    #                print("Warning: Inner solver might not converge.")
    #            else:
    #                sigma = (1+tau*mu)/(1+2*tau*mu+Ldg**2*tau**2)/4
    #                print("Warning: Inner solver might diverge.")
    #
    #            dd = dgFun(yy)
    #            prevNorm = np.linalg.norm(dd)
    #            yy = yy - sigma*dd
    #            dd  = dgFun(yy)
    #            jj = 1
    #            while np.linalg.norm(dd) >= tolInner and jj < nIterInner and np.linalg.norm(dd) < prevNorm:
    #                prevNorm = np.linalg.norm(dd)
    #                yy = yy - sigma*dd
    #                dd  = dgFun(yy)
    #
    #                if printDG>=2:
    #                    print("k: %3i, ||d||: %3.7e" % (jj+1, np.linalg.norm(dd)))
    #                jj+=1
    #
    #            if printDG>=1 and jj == nIterInner:
    #                print("k: %3i, stop inner iter because nIter reached, d: %3.2e" % (jj+1, np.linalg.norm(dd)))
    #            elif printDG>=1 and np.linalg.norm(dd) >= prevNorm:
    #                print("k: %3i, stop inner iter because diff failed to reduce, d: %3.2e" % (jj+1, np.linalg.norm(dd)))
    #            else:
    #                print("k: %3i, stop inner iter, diff: %3.2e < tol: %3.2e." % (jj+1, np.linalg.norm(dd), tolInner))
    #
    #

    # %% Linear, positive definite

    elif Vtype == "linear_pd":
        if dg_type in ["gonzalez", "mean_value"]:
            if type(tau) in [float, np.float64]:
                yy = np.linalg.solve((2 / tau * np.identity(dim) + Vmat), 2 * (Vdata - np.dot(Vmat, yy))) + yy
            else:
                yy = np.linalg.solve((np.diag(2 / tau) + Vmat), 2 * (Vdata - np.dot(Vmat, yy))) + yy

        elif dg_type in ["itoh_abe_r", "itoh_abe_c", "itoh_abe_rc"]:
            if dg_type == "itoh_abe_r":
                jj_vec = np.random.randint(0, dim, dim)
            elif dg_type == "itoh_abe_rc":
                jj_vec = np.random.choice(range(dim), dim, replace=False)
            for ii in range(dim):
                if dg_type in ["itoh_abe_r", "itoh_abe_rc"]:
                    jj = jj_vec[ii]
                else:
                    jj = ii
                if type(tau) in [float, np.float64]:
                    Tau = tau
                else:
                    Tau = tau[jj]
                yy[jj] += (Vdata[jj] - np.dot(yy, Vmat[:, jj])) / (1 / Tau + Vmat[jj, jj] / 2)

        elif dg_type == "itoh_abe_rs":
            for ii in range(dim):
                f = np.random.normal(size=dim)
                f = f / np.linalg.norm(f)
                Qf = np.dot(Vmat, f)
                yy += (np.dot(Vdata, f) - np.dot(Qf, yy)) / (1 / tau + np.dot(Qf, f) / 2) * f

    return yy, Vfun(yy)


# %%


def dg_fixed_point(yy, Vfun, Vyy, gradV, tau, type, coord=None, dd=None):
    tol = 1e13

    if type == "coordinate":

        def f(z):
            if z == 0:
                return tau * gradV(yy)[coord]
            else:
                yyNew = np.array(yy)
                yyNew[coord] += z
                return tau * (Vfun(yyNew) - Vyy) / z + z

    elif type == "search":

        def f(z):
            if z == 0:
                return tau * np.dot(gradV(yy), dd)
            else:
                yyNew = np.array(yy)
                yyNew += z * dd
                return tau * (Vfun(yyNew) - Vyy) / z + z

    zz = 1e0
    sign1 = np.sign(f(zz))
    sign2 = np.sign(f(-zz))
    while (sign1 == sign2) and (sign1 != 0) and zz < tol:
        zz = 10 * zz
        sign1 = np.sign(f(zz))
        sign2 = np.sign(f(-zz))

    if sign1 == sign2:
        return 0
    else:
        ZZ, info = scipy.optimize.brenth(f, -zz, zz, xtol=1e-18, full_output=True)
        print("DOING MY METHOD. Discrep = " + str(f(ZZ)))
        if info.flag != "converged":
            print("Uh Oh!")
        return ZZ


def dg_fixed_point_IA(yy, Vfun_IA, Vyy, gradV_IA, tau, coord=None, dd=None):
    tol = 1e10

    def f(z):
        if z == 0:
            return tau * gradV_IA(yy, coord)
        else:
            yyNew = np.array(yy)
            yyNew[coord] += z
            return tau * (Vfun_IA(yyNew, coord) - Vyy) / z + z

    zz = 1e0
    sign1 = np.sign(f(zz))
    sign2 = np.sign(f(-zz))
    while (sign1 == sign2) and (sign1 != 0) and zz < tol:
        zz = 10 * zz
        sign1 = np.sign(f(zz))
        sign2 = np.sign(f(-zz))

    if sign1 == sign2:
        return 0
    else:
        ZZ, info = scipy.optimize.brenth(f, -zz, zz, xtol=1e-10, full_output=True)
        #        print("DOING MY METHOD. Discrep = " + str(f(ZZ)))
        if info.flag != "converged":
            print("Uh Oh!")
        return ZZ
