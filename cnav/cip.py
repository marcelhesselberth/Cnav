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


def GCRS_to_ITRS(vec_gcrs, UT1_jd2, tjc, xp, yp):
    """
    Full IAU 2006 transform including Polar Wobble.
    xp, yp in radialen.
    """
    # 1. Celestial to Intermediate (Q matrix / Mcio)
    Q = Mcio(tjc)
    
    # 2. Earth Rotation (R3 matrix met ERA)
    R_era = R3(ERA(UT1_jd2))
    
    # 3. Polar Wobble (W matrix)
    W = Mw(xp, yp, tjc)
    
    # Totale rotatie: W * R_era * Q
    # (Merk op dat jouw huidige R() functie R3(ERA) @ Mcio al doet)
    return W @ R_era @ Q @ vec_gcrs


def get_rotation_matrix(mjd_utc, iers_finals, leap_seconds):
    """
    Berekent de volledige IAU 2006 rotatiematrix (W * R * Q) voor een MJD UTC.
    
    mjd_utc: float, Modified Julian Day in UTC
    iers_finals: instantie van je Finals klasse
    leap_seconds: instantie van je Leapseconds klasse
    """
    # 1. Tijdsschalen bepalen
    # TAI = UTC + dAT
    dat = leap_seconds(mjd_utc)
    mjd_tai = mjd_utc + dat / 86400.0
    
    # TT = TAI + 32.184s
    mjd_tt = mjd_tai + 32.184 / 86400.0
    tjc = (mjd_tt - 51544.5) / 36525.0  # TT in Julian Centuries (vanaf J2000 MJD)
    
    # UT1 = UTC + DUT1 (geinterpoleerd uit Finals)
    iers_data = iers_finals(mjd_utc, full=True)
    if iers_data is None:
        # Fallback als data ontbreekt
        dut1, xp, yp, dx, dy = 0, 0, 0, 0, 0
    else:
        dut1 = iers_data['dut1']
        # Omzetten naar radialen: PMx/y zijn in arcsec, dX/dY in msec
        xp = iers_data['pmx'] * AS2RAD
        yp = iers_data['pmy'] * AS2RAD
        dx = (iers_data['dx'] / 1000.0) * AS2RAD
        dy = (iers_data['dy'] / 1000.0) * AS2RAD
    
    mjd_ut1 = mjd_utc + dut1 / 86400.0
    # UT1_jd2 tuple voor je ERA functie: (MJD_integer + MJD0, fractie)
    ut1_jd2 = (int(mjd_ut1) + 2400000.5, mjd_ut1 % 1)

    # 2. De Matrix Componenten
    # Q: Celestial-to-Intermediate (inclusief dX/dY correcties)
    # Je Mcio berekent X, Y uit XYs06. Voor volledige precisie moet je 
    # dX en dY optellen bij de X en Y van het model.
    Q = Mcio_corrected(tjc, dx, dy) 
    
    # R: Earth Rotation
    Rot_ERA = R3(ERA(ut1_jd2))
    
    # W: Polar Wobble
    W = Mw(xp, yp, tjc)
    
    # 3. Totale rotatie: ITRS = W * R * Q * GCRS
    return W @ Rot_ERA @ Q



def get_star_vector_gcrs(mjd_tt, star_data):
    """
    Berekent de GCRS-eenheidsvector van een ster, inclusief parallax.
    star_data: dict met 'ra', 'dec', 'pm_ra', 'pm_dec', 'parallax' (mas)
    """
    # 1. Bereken tijd sinds J2000 (Julian Centuries)
    tjc = (mjd_tt - 51544.5) / 36525.0
    
    # 2. Correctie voor Eigenbeweging (Proper Motion)
    # pm_ra is vaak in ms/yr, pm_dec in mas/yr. Omzetten naar radialen.
    ra = star_data['ra'] * DEG2RAD + (star_data['pm_ra'] * AS2RAD * tjc)
    dec = star_data['dec'] * DEG2RAD + (star_data['pm_dec'] * AS2RAD * tjc)
    
    # 3. Omzetten naar Cartesiaanse eenheidsvector
    cos_dec = np.cos(dec)
    u_star = np.array([np.cos(ra) * cos_dec, np.sin(ra) * cos_dec, np.sin(dec)])
    
    # 4. Parallax correctie (Barycentrisch -> Geocentrisch)
    # d = 1/parallax. Vector_aarde is de positie van de aarde t.o.v. de zon (in AU).
    parallax_rad = (star_data['parallax'] / 1000.0) * AS2RAD
    E = get_earth_pv_barycentric(mjd_tt) # Gebruik VSOP87 of JPL
    
    # De shift door parallax: u_geocentric = u_star - parallax * E
    u_gcrs = u_star - parallax_rad * E
    return u_gcrs / np.linalg.norm(u_gcrs)


# # 1. Haal data
# star = db.get_star("Sirius")
# matrix = get_rotation_matrix(mjd_utc, iers_finals, leap_seconds)

# # 2. Bereken vector in GCRS (inclusief parallax & eigenbeweging)
# # mjd_tt bereken je via je Leapseconds klasse
# v_gcrs = get_star_vector_gcrs(mjd_tt, star)

# # 3. Roteer naar ITRS (Aardse coördinaten)
# v_itrs = matrix @ v_gcrs

# # 4. Omzetten naar Azimuth / Elevatie
# # (v_itrs[0] = X_earth, v_itrs[1] = Y_earth, v_itrs[2] = Z_earth)
# az, el = itrs_to_azel(v_itrs, lat, lon)


# import sqlite3
# import pandas as pd

def create_star_db(source_csv, db_name="stars.db"):
    """
    Maakt een SQLite database van de Hipparcos catalogus (HIP2).
    Filtert op Vmag < 5.0 voor de helderste sterren.
    """
    # Lees de relevante kolommen uit de catalogus
    # HIP: ID, Vmag: Helderheid, RAdeg/DEdeg: Positie, 
    # Plx: Parallax, pmRA/pmDE: Eigenbeweging
    cols = ['HIP', 'Vmag', 'RAdeg', 'DEdeg', 'Plx', 'pmRA', 'pmDE']
    
    # Gebruik pandas om de CSV in te lezen (pas delimiter aan indien nodig)
    df = pd.read_csv(source_csv, usecols=cols)
    
    # Filter op heldere sterren (Vmag < 5.0)
    bright_stars = df[df['Vmag'] < 5.0].copy()
    
    # Maak verbinding met SQLite
    conn = sqlite3.connect(db_name)
    bright_stars.to_sql('stars', conn, if_exists='replace', index=False)
    
    # Index op HIP ID voor snelle lookups
    conn.execute("CREATE INDEX idx_hip ON stars(HIP)")
    conn.close()
    print(f"Database {db_name} aangemaakt met {len(bright_stars)} sterren.")

# Gebruik:
# create_star_db("hip2_full.csv")

# Wees voorzichtig met code.
# 2. Ster-data ophalen voor je pipeline
# In je programma gebruik je dan deze functie om de parameters op te halen voor je get_star_vector_gcrs functie:
# python

def get_star_params(hip_id, db_name="stars.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    query = "SELECT RAdeg, DEdeg, Plx, pmRA, pmDE FROM stars WHERE HIP = ?"
    cursor.execute(query, (hip_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'ra': row[0],      # graden
            'dec': row[1],     # graden
            'parallax': row[2], # milli-arcseconds
            'pm_ra': row[3] / 1000.0,  # omzetten naar arcsec/yr
            'pm_dec': row[4] / 1000.0  # omzetten naar arcsec/yr
        }
    return None


# Interpolate XYs
def generate_chebyshev_coeffs(t_start, t_end, order=12):
    """
    Genereert coëfficiënten voor X, Y en s over een tijdsinterval.
    """
    # 1. Definieer de Chebyshev-punten (nodes) voor maximale precisie
    # Dit voorkomt het fenomeen van Runge (oscillaties aan de randen)
    k = np.arange(order + 1)
    nodes_standard = np.cos(np.pi * (2 * k + 1) / (2 * order + 2))
    
    # 2. Transformeer nodes naar jouw tijdsinterval [t_start, t_end]
    t_nodes = 0.5 * (nodes_standard + 1) * (t_end - t_start) + t_start
    
    # 3. Bereken de 'dure' IAU2006 waarden op deze specifieke punten
    # Gebruik hier je bestaande (Numba) functie
    vals = np.array([XYs06(t) for t in t_nodes]) # Shape: (order+1, 3)
    
    # 4. Fit de Chebyshev coëfficiënten (gebruik bijv. np.polynomial)
    # Dit doe je één keer per dag (of interval)
    coeffs_x = np.polynomial.chebyshev.Chebyshev.fit(t_nodes, vals[:,0], order)
    coeffs_y = np.polynomial.chebyshev.Chebyshev.fit(t_nodes, vals[:,1], order)
    coeffs_s = np.polynomial.chebyshev.Chebyshev.fit(t_nodes, vals[:,2], order)
    
    return coeffs_x, coeffs_y, coeffs_s

# --- GEBRUIK IN JE ALMANAK LOOP ---
# In plaats van XYs06(t) aan te roepen voor elke seconde:
# x_fast = coeffs_x(t) 
# Dit is slechts een handvol vermenigvuldigingen!


