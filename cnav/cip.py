#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:28:19 2024

@author: Marcel Hesselberth
"""


import numpy as np
from constants import AS2RAD, UAS2RAD, PI, PI2, DEG2RAD, JD2000
from rot3d import *
from xys import XYs06 as XYs, PFW06_gamma_phi as PFW
import webdata


"""
Celestial to terrestrial coordinate transformations based on the X, Y
components of the CIP unit vector and on quantity s, the CIO locator.

This code closely follows USNO Circular 179 and Wallace and Capitaine,
Precession-nutation procedures consistent with IAU 2006 resolutions
"""


# ToDo: check units. All angles radians.


# Instantiate helper classes
TAImUTC = webdata.leapseconds()
UT1mUTC  = webdata.finals()
finals = lambda mjd: UT1mUTC(mjd, full=True)


def Mcio(tjc, dX=0, dY=0):
    """
    tjc: time in Julian Centuries (TT) for X, Y, CIO locator s
    dX, dY: IERS corrections (interpolated from finals).
    
    """
    X, Y = XYs(tjc)
    X, Y = X + dX, Y_mod + dY  # IERS feedback

    M = np.empty((3,3))
    Z = np.sqrt(1 - X*X - Y*Y)
    a = 1 / (1+Z)
    ss, cs = np.sin(s), np.cos(s)

    M[0][0] = cs + a * X * ( Y * ss - X * cs)
    M[0][1] = -ss + a * Y * (Y * ss - X * cs)
    M[0][2] = -(X * cs - Y * ss)
    M[1][0] = ss - a * X * (Y * cs + X * ss)
    M[1][1] = cs - a * Y * (Y * cs + X * ss)
    M[1][2] = -(Y * cs + X * ss)
    M[2][0] = X
    M[2][1] = Y
    M[2][2] = Z
    return M

def Mcio_lp(tt_jd):
    tau = tt_jd - JD2000
    Omega = 2.182 - 9.242e-4 * tau  # radians
    X = 2.6603e-7*tau - 33.2e-6 * np.sin(Omega)
    Y = -8.14e-14*tau**2 + 44.6e-6 * np.cos(Omega)
    M = np.array(
        [[1, 0, -X],
         [0, 1, -Y],
         [X, Y, 1]], dtype=np.float)
    return M

def ERA(UT1_jd2):  # Julian UT1 date since JD2000
    Tu_date, Tu_fraction = UT1_jd2
    Du_date, Du_fraction = Tu_date - JD2000, Tu_fraction
    f = Tu_fraction + Tu_date % 1
    Du = Du_fraction + Du_date
    turns = ((f + 0.7790572732640) + 0.00273781191135448 * Du) % 1
    return PI2 * turns


def R(UT1_jd2, tjc, mcio = None):
    if not mcio:
        mcio = Mcio(tjc)
    return R3(ERA(UT1_jd2)) @ mcio


def W(tjc, pmx, pmy):
    """
    tjc: time in Julian Centuries (TT) for the s' locator.
    pmx, pmy: Polar motion coördinates in radians (interpolated from finals).

    """
    # TIO locator s' in radians (approximation).
    sp = -47.0 * UAS2RAD * tjc
    
    # W = R3(s') * R2(xp) * R1(yp)
    return R3(sp) @ R2(pmx) @ R1(pmy)


# classical equinox based NPB matrix and EO
def Mclass_EO(tjc):
    gamma, phi = PFW(tjc)
    k = np.array([np.sin(phi)*np.sin(gamma), \
                  -np.sin(phi)*np.cos(gamma), \
                  np.cos(phi)])
    X, Y, s = XYs(tjc)
    Z = np.sqrt(1 - X*X - Y*Y)
    n = np.array([X, Y, Z])
    nxk = np.cross(n, k)
    nxnxk = np.cross(n, nxk)
    YY = nxk / np.linalg.norm(nxk)
    y = nxnxk / np.linalg.norm(nxnxk)
    Mclass = np.array([YY, y, n], dtype=np.double)
    
    zp1 = 1 + Z
    S = np.array([1 - X * X / zp1, -X * Y / zp1, -X], dtype = np.double)

    return Mclass, s - np.arctan2(np.dot(y, S), np.dot(YY, S))


def Mclass(Mcio, tjc):
    gamma, phi = PFW(tjc)
    k = np.array([np.sin(phi)*np.sin(gamma), \
                  -np.sin(phi)*np.cos(gamma), \
                  np.cos(phi)])
    n = Mcio[2]
    nxk = np.cross(n, k)
    nxnxk = np.cross(n, nxk)
    YY = nxk / np.linalg.norm(nxk)
    y = nxnxk / np.linalg.norm(nxnxk)
    Mclass = np.array([YY, y, n], dtype=np.double)
    return Mclass
 
    
def EO(tjc):
    X, Y, S = XYs(tjc)
    return Mclass_EO(tjc)[1]


def GST(UT1_jd2, eo):
        return ERA(UT1_jd2) - eo




