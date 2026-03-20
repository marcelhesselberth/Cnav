#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:03:57 2026

@author: Marcel Hesselberth
"""


import numpy as np


JD2000  = 2451545.0  # Start of the JD2000 epoch
MJD0    = 2400000.5  # For computing Modified Julian days, mjd's
timescales = ["UTC", "UT1", "TAI", "GPS", "TT", "TDB", "TCG"]
mdays      = {1:31,2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31,
              11:30, 12:31}
wdays      = { 0:"Sun", 1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat" }


def is_gregorian(YYYY:int, MM:int, DD:int) -> bool:
    """
    Check if the date is in the Gregorian calendar.

    Parameters
    ----------
    YYYY : int
           Year from -4712 to 9999
    MM   : int
           Month (1-12)
    DD   : int
           Day between 1 and mdays (and 29-2 in a leap year)

    Returns
    -------
    Bool
        True if the date is in the Gregorian calendar, otherwise False.

    """
    return (YYYY + MM/12) * 365.25 + DD > 578140


def is_leap_year(YYYY:int) -> bool:
    """
    Checks if a year is a leap year (february has 29 days).
    This works both in the Julian and the Gregorian calendar.

    Parameters
    ----------
    YYYY : Int
        Year from -4712 to 9999

    Returns
    -------
    leapyear : Boolean
        True if YYYY is a leap year.

    """
    if is_gregorian(YYYY, 2, 28):
        if YYYY % 4 == 0:               # possibly leap
            if YYYY % 400 == 0:         # leap
                leapyear = True
            else:
                if YYYY % 100 == 0:     # common
                    leapyear = False
                else:                   # leap
                    leapyear = True
        else:
            leapyear = False
    else:                               # julian
        if YYYY % 4 == 0:               # julian leap year
            leapyear = True
        else:
            leapyear = False
    return leapyear


def JD(YYYY:int, MM:int, DD:int) -> float:
    """
    Julian day number. Assumes that year, month and day are valid.
    Works for all positive JD (years from -4712).


    Parameters
    ----------
    YYYY : int
           Year.
    MM   : int
           Month.
    DD   : int
           Day.

    Returns
    -------
    float
           The corresponding Julian day number

    """
    if YYYY < -4712:
        raise(ValueError("Invalid date: %d-%d-%d" % (YYYY, MM, DD)))
    if MM <= 2:
        Y  = YYYY - 1
        M  = MM + 12
    else:
        Y = YYYY
        M = MM
    A = int(Y/100)
    if (YYYY + MM/12) * 365.25 + DD < 578140:
        B = 0
    else:
        B = 2 - A + int(A/4)
    return int(365.25*(Y+4716)) + int(30.6001*(M+1)) + DD + B - 1524.5


def MJD(YYYY:int, MM:int, DD:int) -> float:
    """
    Modified Julian day of a valid date.

    Parameters
    ----------
    YYYY : int
           Year
    MM   : int
           Month
    DD   : int
           Day

    Returns
    -------
    float
        Modified Julian day for the geven date

    """
    return JD(YYYY, MM, DD) - 2400000.5


def RJD(jd:float) -> (int, int, int, float):
    """
    Reverse Julian Day. Compute date (YYYY, MM, DD) from jd.

    Parameters
    ----------
    jd : float
         Julian day jd. jd must be positive,

    Returns
    -------
    YYYY : int
           Year.
    MM   : int
           Month.
    DD   : int
           Day.
    F    : float
           Day fraction.
    """
    jd5 = jd + 0.5
    Z   = int(jd5)
    F   = jd5 - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)  # positive
        A = Z + 1 + alpha - int(alpha/4)
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    DD = B - D - int(30.6001 * E)
    if E < 14:
        MM = E - 1
    else:
        MM = E - 13
    assert(MM >=1 and MM <= 12)
    if MM > 2:
        YYYY = C - 4716
    else:
        YYYY = C - 4715
    return YYYY, MM, DD, F

def RMJD(mjd):
    """
    Reverse Modified Julian Day. Compute date (YYYY, MM, DD) from jd.

    Parameters
    ----------
    jd : float
         Julian day jd. jd must be positive,

    Returns
    -------
    YYYY : int
           Year.
    MM   : int
           Month.
    DD   : int
           Day.
    F    : float
           Day fraction.
    """
    return(RJD(mjd+2400000.5))


def TJC(tt, tt2=0.0):
    """
    Julian centuries since JD2000. Ised in IERS transform.


    Parameters
    ----------
    tt   : int / float
           Year.
    tt2  : float

    Returns
    -------
    float
           The corresponding Julian century
    """
    return ((tt - 2451545.0) + tt2) / 36525


def BY(TT_JD):
    """
    Besselian year.

    Parameters
    ----------
    TT_JD   : float

    Returns
    -------
    float
           The corresponding Besselian year
    """
    B = 1900.0 + (TT_JD - 2415020.31352) / 365.242198781


# 
def weekday_nr(jd):    # jd is a julian day number without time
    """
    Weekday number

    Parameters
    ----------
    JD   : float

    Returns
    -------
    int
           The weekday number. 0 is sunday, 1 is monday etc.
    """
    return int((jd+1.5)) % 7


def weekday_str(jd):
    """
    Three character day string

    Parameters
    ----------
    JD   : float

    Returns
    -------
    string
           The weekday string of JD.
    """
    return wdays[weekday_nr(jd)]


def TTmTDB(tt_jd):
    """
    Lower precision version of TTmTDB, precision 10 us

    Parameters
    ----------
    tt_jd   : float

    Returns
    -------
    float
           TT - TDB in seconds
    """
    T = (tt_jd - JD2000) / 36525
    return  - 0.001657 * np.sin (628.3076 * T + 6.2401)  \
            - 0.000022 * np.sin (575.3385 * T + 4.2970)  \
            - 0.000014 * np.sin (1256.6152 * T + 6.1969) \
            - 0.000005 * np.sin (606.9777 * T + 4.0212)  \
            - 0.000005 * np.sin (52.9691 * T + 0.4444)   \
            - 0.000002 * np.sin (21.3299 * T + 5.5431)   \
            - 0.000010 * T * np.sin (628.3076 * T + 4.2490)

