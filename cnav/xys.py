#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:04:34 2024

@author: Marcel Hesselberth
"""


import numpy as np
from cnumba import cnjit
from constants import UAS2RAD, AS2RAD
import iersch5

# Silence numba dot product warning if arrays are not contiguous
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# =============================================================================
# Data for the IAU2006 transformations, following Capitaine & Wallace.
# =============================================================================

# For evaluating 4th and 5th degree polynomials
powers_4 = np.array([0, 1, 2, 3, 4])
powers_5 = np.array([0, 1, 2, 3, 4, 5])


# Coefficients from nutation theory, consistent with IAU2006.
# See USNO Circular https://aa.usno.navy.mil/downloads/Circular_179.pdf
# and IERS convention 2010, ch.5.
planets    = [[908103.259872,  538101628.688982],
              [655127.283060,  210664136.433548],
              [361679.244588,  129597742.283429],
              [1279558.798488, 68905077.493988 ],
              [123665.467464,  10925660.377991 ],
              [180278.799480,  4399609.855732  ],
              [1130598.018396, 1542481.193933  ],
              [1095655.195728, 786550.320744   ]]
p_a         = [5028.8200,      1.112022        ]
luni_solar = [[485868.249036, 1717915923.2178, 31.8792,  0.051635, -0.00024470],
              [1287104.79305, 129596581.0481,  -0.5532,  0.000136, -0.00001149],
              [335779.526232, 1739527262.8478,-12.7512, -0.001037,  0.00000417],
              [1072260.70369, 1602961601.2090, -6.3706,  0.006593, -0.00003169],
              [450160.398036, -6962890.5431,    7.4722,  0.007702, -0.00005939]]


# Fukushima-Williams coefficients for gamma and phi consistent with IAU2006.
# For equinox based transforms.
# IERS Conventions (2010) in Table 5.2e
# Capitaine, N., Wallace, P.T., & Chapront, J. (2003)
PFW_poly_5 = [[   -0.052928,   10.556378,  0.4932044, -0.00031238, -0.000002788,  0.0000000260],
              [84381.412819,  -46.811016,  0.0511268,  0.00053289, -0.000000440, -0.0000000176]]

PFW_poly_5 = np.array(PFW_poly_5, dtype = np.double).T


# Rearrange for calculating the arguments of the IERS2006 non-polynomial part
arg_1_5  = np.array(luni_solar, dtype = np.double).T
arg_6_13 = np.array(planets, dtype = np.double).T
arg_14   = np.array(p_a, dtype = np.double).T


# =============================================================================
# Private functions and classes
# =============================================================================

@cnjit(signature_or_function='f8[:](f8)' , cache=True)
def Phi(tjc):
    """
    The fundamental arguments of nutation theory consistent with IAU2006.
    A[0:5] are the luni-solar arguments l, l', F, D, Om, degree 0-4.
    A[5:13] are the planetary terms (heliocentric ecliptic longitudes 
    of the planets Mercury through Neptune), L_Me, L_Ve,  L_E, L_Ma,
    L_J, L_Sa, L_U, L_Ne, degree 0-1.
    A[14] is an approximation of the precession in longitude p_A.

    Parameters
    ----------
    tjc : float
          TDB time in julian centuries. Using TT causes negligible error.

    Returns
    -------
    A : np.array(14, dtype=float)
    """
    A = np.empty(14, dtype=np.double)
    T = np.power(tjc, powers_4)
    A[0:5]  = np.dot(T, arg_1_5)  # calculate polynomials
    A[5:13] = np.dot(T[0:2], arg_6_13)
    A[13]   = np.dot(T[1:3], arg_14)
    A *= AS2RAD
    return np.ascontiguousarray(A)


class XYs06a:
    """Packages the IERS data with the numerical algorithms"""

    def __init__(self, polynomial, arrays):
        self.poly = polynomial
        self.arrays = arrays


    @cnjit(signature_or_function='f8(f8[:], f8[:], i4, f8[:], f8[:], f8[:, :])')
    def nonpoly(T, phi, j, S, C, M):
        ARG = np.dot(phi, M)  # Phi_i = sum over j Mij 
        terms = ( S * np.sin(ARG) + C * np.cos(ARG) ) * T[j]
        return np.sum(terms)
        

    def __call__(self, T, phi):
        result = 0.0
        for j, nSCM in self.arrays.items():  # for each order j
            n, S, C, M = nSCM  # number of terms, sin, cos, Mij
            result += XYs06a.nonpoly(T, phi, j, S, C, M)
        result += np.dot(T, self.poly)  # add the polynomial part
        return result


# Instantiate helper classes
X           = XYs06a(iersch5.X_polynomial_5,     iersch5.X)
Y           = XYs06a(iersch5.Y_polynomial_5,     iersch5.Y)
spXY2       = XYs06a(iersch5.spXY2_polynomial_5, iersch5.spXY2)


# =============================================================================
# Functions provided by this module
# =============================================================================

def XYs06(tjc):
    """
    Compute the CIP X,Y coordinates and the value of the CIO locator s.

    The time is given in Julian centuries since JD2000.

    Parameters
    ----------
    tjc : float
          Time in Julian centuries (see dtmath.TJC) TDB (or TT).

    Returns
    -------
    result : np.ndarray(dtype=np.double)
             [X, Y, s]

    """
    phi = Phi(tjc)
    T = np.power(tjc, powers_5)
    x = X(T, phi)
    y = Y(T, phi)
    spxy2 = spXY2(T, phi)
    result = np.array([x, y, spxy2], dtype = np.double)
    result *= UAS2RAD
    result[2] = result[2] - (result[0] * result[1]) / 2  # s = spXY2 - XY/2
    return result


def PFW06_gamma_phi(tjc):
    """
    Fukushima Williams angles gamma and phi.
    When the CIP based approach is used but equinox /GST based data must (also)
    be computed, only these two angles are required.

    Parameters
    ----------
    tjc : float
          Time in Julian centuries (see dtmath.TJC) TDB (or TT).

    Returns
    -------
    result : np.ndarray(dtype=np.double)
             [gamma, phi]

    """
    return np.dot(np.power(tjc, powers_5), PFW_poly_5) * AS2RAD
