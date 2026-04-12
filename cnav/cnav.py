#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:52:34 2026

@author: hessel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modern_nav.py — Offline modern celestial navigation toolkit
Author: Marcel Hesselberth
Purpose: Compute LOPs, multi-sight fixes, noon sights, Polaris latitude fully offline
Dependencies: numpy, scipy, cip.py, xys.py
"""

import numpy as np
from cip import Mcio, R, ERA
from xys import XYs06
from datetime import datetime
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. Full 57 navigational stars (J2000, RA hours, Dec degrees)
# ----------------------------------------------------------------------
star_data = [
    ("Sirius", 6.75248, -16.7161),
    ("Canopus", 6.3992, -52.6957),
    ("Arcturus", 14.261, 19.1825),
    ("Vega", 18.6156, 38.7837),
    ("Capella", 5.2782, 45.998),
    ("Rigel", 5.2423, -8.2016),
    ("Procyon", 7.655, 5.225),
    ("Achernar", 1.6286, -57.2367),
    ("Betelgeuse", 5.9195, 7.407),
    ("Hadar", 14.0637, -60.3730),
    ("Altair", 19.8464, 8.8683),
    ("Aldebaran", 4.5987, 16.5093),
    ("Antares", 16.4901, -26.4319),
    ("Spica", 13.4199, -11.1614),
    ("Pollux", 7.7553, 28.0262),
    ("Fomalhaut", 22.9608, -29.6222),
    ("Deneb", 20.6905, 45.2803),
    ("Regulus", 10.1395, 11.9672),
    ("Adhara", 6.9771, -28.9721),
    ("Castor", 7.5767, 31.8883),
    ("Gacrux", 12.5194, -57.1132),
    ("Bellatrix", 5.4189, 6.3497),
    ("Elnath", 5.4381, 28.6075),
    ("Miaplacidus", 9.2192, -69.7172),
    ("Alnilam", 5.6036, -1.2019),
    ("Alnair", 22.1370, -46.9608),
    ("Alioth", 12.9004, 55.9598),
    ("Menkent", 14.1111, -36.3697),
    ("Suhail", 14.6601, -43.9994),
    ("Dubhe", 11.0621, 61.7508),
    ("Mirfak", 3.4054, 49.8611),
    ("Wezen", 7.0767, -26.3932),
    ("Sargas", 17.6230, -42.9975),
    ("Kaus Australis", 18.5366, -34.3842),
    ("Avior", 8.3759, -59.5097),
    ("Alkaid", 13.7923, 49.3133),
    ("Mimosa", 12.7953, -59.6888),
    ("Alhena", 6.7525, 16.3990),
    ("Regor", 7.1099, -63.0993),
    ("Algol", 3.1366, 40.9556),
    ("Hamal", 2.1190, 23.4628),
    ("Polaris", 2.5303, 89.2641),
    ("Kochab", 14.8451, 74.1555),
    ("Phecda", 11.8972, 53.6948),
    ("Merak", 11.0307, 56.3824),
    ("Alpheratz", 0.1398, 29.0904),
    ("Mirach", 1.9230, 35.6206),
    ("Algorab", 12.0253, -16.0517),
    ("Sualocin", 21.0301, 15.1822),
    ("Rotanev", 21.0920, 14.5726),
    ("Enif", 21.7369, 9.8750),
    ("Markab", 23.0407, 15.2053),
    ("Scheat", 23.4218, 28.0836),
    ("Algenib", 0.1396, 15.1723)
]

stars = []
for name, ra_hr, dec_deg in star_data:
    stars.append({
        "name": name,
        "ra_rad": np.deg2rad(ra_hr * 15),
        "dec_rad": np.deg2rad(dec_deg)
    })

# ----------------------------------------------------------------------
# 2. Time conversions
# ----------------------------------------------------------------------
def jd_from_datetime(dt):
    year, month, day = dt.year, dt.month, dt.day
    hr, mn, sec = dt.hour, dt.minute, dt.second + dt.microsecond/1e6
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25*(year+4716)) + int(30.6001*(month+1)) + day + B - 1524.5
    JD += (hr + mn/60 + sec/3600) / 24
    return JD

def utc_to_ut1_tt(utc_datetime, dut1_sec, delta_t_sec):
    JD = jd_from_datetime(utc_datetime)
    UT1 = JD + dut1_sec / 86400.0
    TT  = JD + delta_t_sec / 86400.0
    return UT1, TT

# ----------------------------------------------------------------------
# 3. Celestial GCRS vector
# ----------------------------------------------------------------------
def celestial_gcrs_vector(body, tjc, tt):
    if "ra_rad" in body:
        ra, dec = body["ra_rad"], body["dec_rad"]
    else:
        ra, dec = INPOP_position(body["name"], tt)  # placeholder
    return np.array([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)])

# ----------------------------------------------------------------------
# 4. GCRS → Earth-fixed
# ----------------------------------------------------------------------
def gcrs_to_tirs(r_gcrs, ut1_2000, tjc):
    mcio = Mcio(tjc)
    return R(ut1_2000, tjc, mcio) @ r_gcrs

# ----------------------------------------------------------------------
# 5. Topocentric conversion
# ----------------------------------------------------------------------
def topocentric_altaz(r_ef, lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    R_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])
    r_enu = R_enu @ r_ef
    e, n, u = r_enu
    h = np.arcsin(u)
    A = np.arctan2(e, n) % (2*np.pi)
    return np.rad2deg(h), np.rad2deg(A)

# ----------------------------------------------------------------------
# 6. LOP intercept
# ----------------------------------------------------------------------
def intercept(h_obs, h_calc, azimuth):
    d_nm = (h_obs - h_calc) * 60.0
    return d_nm, azimuth

# ----------------------------------------------------------------------
# 7. Sight reduction
# ----------------------------------------------------------------------
def reduce_sight(body, utc_datetime, h_obs, lat_deg, lon_deg, dut1_sec, delta_t_sec):
    UT1, TT = utc_to_ut1_tt(utc_datetime, dut1_sec, delta_t_sec)
    tjc = (TT - 2451545.0)/36525.0
    ut1_2000 = UT1 - 2451545.0
    r_gcrs = celestial_gcrs_vector(body, tjc, TT)
    r_ef = gcrs_to_tirs(r_gcrs, ut1_2000, tjc)
    h_calc, az_calc = topocentric_altaz(r_ef, lat_deg, lon_deg)
    d, az = intercept(h_obs, h_calc, az_calc)
    return {"h_calc": h_calc, "az_calc": az_calc, "LOP_nm": d, "LOP_az": az}

# ----------------------------------------------------------------------
# 8. Multi-sight fix
# ----------------------------------------------------------------------
def fix_from_sights(sights, lat_guess, lon_guess):
    R_nm = 3440.065
    def cost(x):
        lat, lon = np.deg2rad(x[0]), np.deg2rad(x[1])
        cost_sum = 0.0
        for s in sights:
            az = np.deg2rad(s["LOP_az"])
            d = s["LOP_nm"]
            delta_lat = (d/R_nm)*np.cos(az)
            delta_lon = (d/R_nm)*np.sin(az)/np.cos(lat)
            cost_sum += delta_lat**2 + delta_lon**2
        return cost_sum
    res = minimize(cost, [lat_guess, lon_guess])
    return res.x[0], res.x[1]

# ----------------------------------------------------------------------
# 9. Noon sight latitude
# ----------------------------------------------------------------------
def latitude_noon(h_obs, declination_deg):
    return 90.0 - h_obs + declination_deg

# ----------------------------------------------------------------------
# 10. Polaris latitude
# ----------------------------------------------------------------------
def latitude_polaris(h_obs):
    return h_obs + 0.7

# ----------------------------------------------------------------------
# 11. Offline UT1 / ΔT table
# ----------------------------------------------------------------------
ut_table = [
    [2024, 1, 0.334, 69.0],
    [2024, 6, 0.345, 69.0],
    [2025, 1, 0.356, 69.1],
    [2025, 6, 0.367, 69.1],
    [2026, 1, 0.378, 69.2],
    [2026, 6, 0.389, 69.2],
]

def lookup_ut1_deltat(year, month):
    table = np.array(ut_table)
    years = table[:,0]
    months = table[:,1]
    dut1s = table[:,2]
    deltas = table[:,3]
    dec_month = year + (month-1)/12.0
    dec_table = years + (months-1)/12.0
    if dec_month <= dec_table[0]:
        return dut1s[0], deltas[0]
    if dec_month >= dec_table[-1]:
        return dut1s[-1], deltas[-1]
    for i in range(len(dec_table)-1):
        if dec_table[i] <= dec_month <= dec_table[i+1]:
            f = (dec_month - dec_table[i]) / (dec_table[i+1]-dec_table[i])
            dut1 = dut1s[i] + f*(dut1s[i+1]-dut1s[i])
            delta_t = deltas[i] + f*(deltas[i+1]-deltas[i])
            return dut1, delta_t

# ----------------------------------------------------------------------
# 12. Running fix example
# ----------------------------------------------------------------------
def running_fix_example():
    observations = [
        ("Sirius", 25.3),
        ("Vega", 42.7),
        ("Capella", 33.8)
    ]
    lat_guess, lon_guess = 36.0, -122.0
    utc = datetime(2026, 3, 15, 22, 0, 0)
    dut1, delta_t = lookup_ut1_deltat(utc.year, utc.month)
    sights = []
    for body_name, h_obs in observations:
        star = next(s for s in stars if s["name"] == body_name)
        sight = reduce_sight(star, utc, h_obs, lat_guess, lon_guess, dut1, delta_t)
        sights.append(sight)
    lat_fix, lon_fix = fix_from_sights(sights, lat_guess, lon_guess)
    print("\nRunning fix from multiple sights:")
    for s, obs in zip(sights, observations):
        print(f"{obs[0]}: alt {s['h_calc']:.2f}°, az {s['az_calc']:.2f}°, LOP {s['LOP_nm']:.2f} nm")
    print(f"Estimated fix: lat {lat_fix:.4f}°, lon {lon_fix:.4f}°")

# ----------------------------------------------------------------------
# 13. Run example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    running_fix_example()
    
    
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
