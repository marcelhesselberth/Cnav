#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:36:00 2026

@author: Marcel Hesselberth
"""


from inpop import Inpop
import numpy as np


class Observer(Inpop):
    def __init__(self, filename=None, load=None):
        super().__init__(filename, load)
        c = self.constants
        self.gm_map = {10: c["GM_Sun"], 0: c["GM_Mer"], 1: c["GM_Ven"],
                       4: c["GM_Jup"], 5: c["GM_Sat"], 6: c["GM_Ura"],
                       7: c["GM_Nep"]}
        self.clight_au_day = c["CLIGHT"] / self.AU * 86400

    def PV_apparent_general(self, jd, t, c, max_iter=10, tol=1e-12, **kwargs):
            """
            General apparent position (Light-time + Deflection + Aberration).
            """
            clight = self.clight_au_day
            
            # 1. Fix observer state at time jd
            pv_c_full = self.PV(jd, c, 11, rate=True, **kwargs)
            pos_obs = pv_c_full[0]  # Position vector (3,)
            vel_obs = pv_c_full[1]  # Velocity vector (3,)
    
            # 2. Iterative Light-Time Correction
            tau = 0.0
            for _ in range(max_iter):
                # Explicitly take the position row [0]
                pos_t_ssb = self.PV(jd - tau, t, 11, rate=False, **kwargs)[0]
                pos_geom = pos_t_ssb - pos_obs
                
                dist = np.linalg.norm(pos_geom)
                new_tau = dist / clight
                if abs(new_tau - tau) < tol:
                    break
                tau = new_tau
        
            # 3. Relativistic Gravitational Deflection
            u = pos_geom / dist
            deflection = np.zeros(3)
            
            # Ensure t and c are integers for comparison
            t_id = t if isinstance(t, int) else self.bodycodes[t.lower()]
            c_id = c if isinstance(c, int) else self.bodycodes[c.lower()]
    
            for body_idx, mu in self.gm_map.items():
                if body_idx == t_id or body_idx == c_id: continue
                
                # Position of deflecting body relative to observer
                pos_body = self.PV(jd, body_idx, 11, rate=False, **kwargs)[0]
                r_b = pos_body - pos_obs
                dist_b = np.linalg.norm(r_b)
                q = r_b / dist_b
                
                dot_uq = np.dot(u, q) # This is now a scalar
                if 1.0 + dot_uq > 1e-10:
                    factor = (4.0 * mu) / (clight**2 * dist_b * (1.0 + dot_uq))
                    deflection += factor * (np.cross(q, np.cross(u, q)))
        
            u_deflected = u + deflection
            u_deflected /= np.linalg.norm(u_deflected)
        
            # 4. Stellar Aberration (Rigorous Lorentz)
            beta = vel_obs / clight
            inv_gamma = np.sqrt(1.0 - np.dot(beta, beta))
            
            denom = 1.0 + np.dot(u_deflected, beta)
            u_apparent = (inv_gamma * u_deflected + (1.0 + np.dot(u_deflected, beta) / (1.0 + inv_gamma)) * beta) / denom
        
            return u_apparent * dist
    
    
    def PV_apparent_earth(self, jd, t, **kwargs):
        """
        Lightweight version: Geocentric (c=2), Sun-only deflection.
        """
        clight = self.clight_au_day
        
        # Ensure t is an integer for the ID check
        t_id = t if isinstance(t, int) else self.bodycodes[t.lower()]
    
        # Get Earth's State relative to SSB (c=2, center=11)
        # Returns [Pos, Vel]
        pv_earth_full = self.PV(jd, 2, 11, rate=True, **kwargs)
        pos_earth = pv_earth_full[0]
        vel_earth = pv_earth_full[1]
    
        # 1. Fast Light-Time (2 iterations)
        tau = 0.0
        for _ in range(2):
            # Target position relative to SSB at retarded time
            pos_t = self.PV(jd - tau, t, 11, rate=False, **kwargs)[0]
            pos_geom = pos_t - pos_earth
            tau = np.linalg.norm(pos_geom) / clight
        
        dist = np.linalg.norm(pos_geom)
        u = pos_geom / dist
    
        # 2. Sun-only Deflection (Body 10)
        # Skip if target is the Sun
        if t_id != 10:
            pos_sun = self.PV(jd, 10, 11, rate=False, **kwargs)[0]
            r_s = pos_sun - pos_earth
            dist_s = np.linalg.norm(r_s)
            q = r_s / dist_s
            
            dot_uq = np.dot(u, q)
            # 4 * GM / (c^2 * r * (1 + cos)) * (q x (u x q))
            # GM_Sun = self.gm_map[10]
            deflection = (4.0 * self.gm_map[10] / (clight**2 * dist_s * (1.0 + dot_uq))) * (np.cross(q, np.cross(u, q)))
            u = (u + deflection)
            u /= np.linalg.norm(u)
    
        # 3. Aberration (Earth velocity relative to SSB)
        beta = vel_earth / clight
        # Using the standard first-order addition for speed
        u_app = (u + beta) / (1.0 + np.dot(u, beta)) 
        
        return u_app * dist


    def to_spherical(self, pos_vec):
        """Converts AU vector to RA (hours), Dec (deg), and Distance (AU)"""
        dist = np.linalg.norm(pos_vec)
        ra = np.arctan2(pos_vec[1], pos_vec[0]) * 12 / np.pi
        dec = np.arcsin(pos_vec[2] / dist) * 180 / np.pi
        return ra % 24, dec, dist

i = Inpop()
o = Observer()

print(i.constants)

# Open a small INPOP file (auto-download if not present)
inpop = Inpop("inpop21a_TDB_m100_p100_tt.dat")

# Inspect available time range and timescale
print(inpop.jd_beg, inpop.jd_end)  # Julian dates
print(inpop.timescale)             # TDB or TCB

# Compute state vector: moon relative to earth at J2000.0
t = 2451545.0 + 25 * 365
pv = inpop.PV(t, 'saturn', 'earth')
print(pv)
print(o.PV_apparent_general(t, 'saturn', 'earth'))
print(o.PV_apparent_earth(t, 'saturn'))
print(o.to_spherical(o.PV_apparent_general(t, "saturn", "earth")))

from skyfield.api import load

# 1. Load ephemeris and timescale
planets = load('de440.bsp')  # Use a recent JPL file for baseline
ts = load.timescale()

# 2. Define Time (Ensure this is TDB to match your Inpop JD)
# 2451545.0 is J2000.0 TDB
tt = ts.tdb_jd(t)

# 3. Define Earth and Target
earth = planets['earth']
mars = planets['saturn barycenter']

# 4. Calculate Apparent Position
# .observe() handles light-time
# .apparent() handles aberration and gravitational deflection
astrometric = earth.at(tt).observe(mars)
apparent = astrometric.apparent()

# 5. Extract raw XYZ in AU
xa, ya, za = astrometric.xyz.au
x, y, z = apparent.xyz.au
ra, dec, dist = apparent.radec()
ra = ra.hours

print(f"Skyfield Astrometric XYZ (AU): [{xa:.12f}, {ya:.12f}, {za:.12f}]")
print(f"Skyfield Apparent    XYZ (AU): [{x :.12f}, {y :.12f}, {z :.12f}]")
print(ra, dec.degrees, dist.au)

def Apparent(self, jd, t, c=2, **kwargs):
    """
    Unified entry point for apparent position. 
    Handles light-time, gravitational deflection, and stellar aberration.
    """
    # Resolve names to IDs
    t_id = t if isinstance(t, int) else self.bodycodes[t.lower()]
    c_id = c if isinstance(c, int) else self.bodycodes[c.lower()]
    
    # 1. Use the optimized Earth-observer path if c is Earth (2)
    if c_id == 2:
        return self.PV_apparent_earth(jd, t_id, **kwargs)
    
    # 2. Use the high-precision General path for any other center
    # This ensures accuracy for Mars-to-Jupiter or Topocentric-to-Moon
    return self.PV_apparent_general(jd, t_id, c_id, **kwargs)

print(o.constants["CLIGHT"] / o.AU * 24 * 60 * 60)

def PV_apparent_general(self, jd, t, c, max_iter=10, tol=1e-12, **kwargs):
    clight = self.clight_au_day
    
    # 1. Waarnemer (Aarde) op tijdstip jd
    pv_c_full = self.PV(jd, c, 11, rate=True, **kwargs)
    pos_obs = pv_c_full[0]
    vel_obs = pv_c_full[1]

    # 2. Aangepaste Lichttijd voor Sterren vs Planeten
    if self.is_star(t):
        # VOOR STERREN: Gebruik de sterpositie op jd (catalogus-standaard)
        # De enige 'tau' die we corrigeren is de positie van de waarnemer
        pos_t_ssb = self.PV(jd, t, 11, rate=False, **kwargs)[0]
        pos_geom = pos_t_ssb - pos_obs
        dist = np.linalg.norm(pos_geom)
        # Geen iteratie nodig voor de ster zelf!
    else:
        # VOOR PLANETEN: Jouw huidige iteratieve loop (jd - tau)
        tau = 0.0
        for _ in range(max_iter):
            pos_t_ssb = self.PV(jd - tau, t, 11, rate=False, **kwargs)[0]
            pos_geom = pos_t_ssb - pos_obs
            dist = np.linalg.norm(pos_geom)
            new_tau = dist / clight
            if abs(new_tau - tau) < tol: break
            tau = new_tau

    # 3. Deflectie en 4. Aberratie (BLIJVEN HETZELFDE)
    # ... rest van je code ...
    
    
def PV_apparent_general(self, jd, t, c, max_iter=10, tol=1e-12, **kwargs):
    clight = self.clight_au_day
    
    # 1. Fix waarnemer op tijdstip jd
    pv_c_full = self.PV(jd, c, 11, rate=True, **kwargs)
    pos_obs = pv_c_full[0]
    vel_obs = pv_c_full[1]

    # --- DE AANPASSING ---
    # Controleer of 't' een ster is (bijv. als het een string is of niet in planetenlijst)
    is_star = isinstance(t, str) or (isinstance(t, int) and t > 100) # Pas aan naar jouw ID-systeem

    if is_star:
        # VOOR STERREN: Geen iteratie. Gebruik de positie op t0 (nu).
        # De eigenbeweging in de catalogus is al 'waargenomen'.
        pos_t_ssb = self.PV(jd, t, 11, rate=False, **kwargs)[0]
        pos_geom = pos_t_ssb - pos_obs
        dist = np.linalg.norm(pos_geom)
        # We slaan de 'for _ in range(max_iter)' loop volledig over.
    else:
        # VOOR PLANETEN: Behoud de iteratieve loop voor jd - tau
        tau = 0.0
        for _ in range(max_iter):
            pos_t_ssb = self.PV(jd - tau, t, 11, rate=False, **kwargs)[0]
            pos_geom = pos_t_ssb - pos_obs
            dist = np.linalg.norm(pos_geom)
            new_tau = dist / clight
            if abs(new_tau - tau) < tol:
                break
            tau = new_tau
    # ---------------------

    # 3. Gravitatieafbuiging (Blijft hetzelfde: dit gebeurt lokaal)
    # ... rest van je code (deflection & aberration) ...
    
class CelestialBody:
    def PV_apparent_general(self, jd, c, ...):
        # De logica die voor BEIDE geldt (Deflectie, Aberratie)
        # Gebruikt self.PV(...) en self.is_star
        pass

class Planet(CelestialBody):
    def PV(self, jd, ...):
        # JPL/SPICE logica
        return pos_jpl
    is_star = False

class Star(CelestialBody):
    def PV(self, jd, ...):
        # Lineaire eigenbeweging + Parallax
        return pos_star
    is_star = True
    


class CelestialObject:
    def __init__(self):
        self.clight_au_day = 173.1446326846693  # Exacte waarde c in AU/dag
        self.gm_sun = 0.0002959122082855911    # Voorbeeld GM_Sun in AU^3/dag^2

    def PV_apparent_earth(self, jd, target_id, **kwargs):
        clight = self.clight_au_day
        is_star = self._check_if_star(target_id)

        # 1. Toestand van de Aarde (t.o.v. SSB)
        pv_earth = self.get_ssb_pv(jd, 399) # 399 = Earth
        pos_earth, vel_earth = pv_earth[0], pv_earth[1]

        # 2. Positie Target (t.o.v. SSB)
        if is_star:
            # STER: Geen iteratie, geen jd-tau. 
            # De eigenbeweging is al 'geobserveerd'.
            pos_t = self.get_ssb_pv(jd, target_id, rate=False)
            pos_geom = pos_t - pos_earth
        else:
            # PLANEET: 2 iteraties volstaan voor navigatie-precisie
            tau = 0.0
            for _ in range(2):
                pos_t = self.get_ssb_pv(jd - tau, target_id, rate=False)
                pos_geom = pos_t - pos_earth
                tau = np.linalg.norm(pos_geom) / clight
        
        dist = np.linalg.norm(pos_geom)
        u = pos_geom / dist

        # 3. Lichtafbuiging door de Zon (alleen als target niet de Zon is)
        if target_id != 10:
            pos_sun = self.get_ssb_pv(jd, 10, rate=False)
            r_s = pos_sun - pos_earth
            d_s = np.linalg.norm(r_s)
            q = r_s / d_s
            
            dot_uq = np.dot(u, q)
            # Voorkom deling door nul bij sterren precies achter de zon
            if (1.0 + dot_uq) > 1e-7:
                factor = (4.0 * self.gm_sun) / (clight**2 * d_s * (1.0 + dot_uq))
                u = u + factor * (q * (1.0 + dot_uq) - u * dot_uq) # Efficiëntere cross-cross vervanging
                u /= np.linalg.norm(u)

        # 4. Aberratie (Standaard eerste orde)
        beta = vel_earth / clight
        u_app = (u + beta) / (1.0 + np.dot(u, beta))
        u_app /= np.linalg.norm(u_app)
        
        return u_app * dist

    def _check_if_star(self, target_id):
        # Implementeer hier je herkenning (bijv. string of hoog ID)
        return isinstance(target_id, str) or target_id > 1000
    
    
import numpy as np

class Star(CelestialObject):
    def __init__(self, row, reference_jd=2448349.0625):
        """
        row: Een Series of dict uit je 'stars.csv'
        reference_jd: Standaard Hipparcos J1991.25
        """
        super().__init__()
        self.reference_jd = reference_jd
        
        # Basisgegevens uit je CSV (in radialen)
        self.ra0 = row['ra_rad']
        self.dec0 = row['dec_rad']
        self.parallax = row['pi_rad']
        
        # Eigenbeweging (rad/dag)
        self.pm_ra_cosdec = row['mu_alpha_cosdec_rad_d']
        self.pm_dec = row['mu_delta_rad_d']
        
        # Voorbereiden van de basisvector op epoch t0
        self.v0 = self._coords_to_vector(self.ra0, self.dec0)

    def _coords_to_vector(self, ra, dec):
        """Zet RA/Dec om naar een eenheidsvector."""
        return np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ])

    def get_ssb_pv(self, jd, target_id=None, rate=False):
        """
        Berekent de positie van de ster t.o.v. SSB op tijdstip jd.
        Volgt de IAU/Hipparcos standaard (lineair).
        """
        dt = jd - self.reference_jd
        
        # 1. Update positie door eigenbeweging (lineaire benadering voor kleine hoeken)
        # Voor extreem hoge precisie over eeuwen heen zou je dit via 
        # differentiële calculus moeten doen, maar voor navigatie volstaat dit:
        d_ra = (self.pm_ra_cosdec / np.cos(self.dec0)) * dt
        d_dec = self.pm_dec * dt
        
        # Nieuwe eenheidsvector u
        u = self._coords_to_vector(self.ra0 + d_ra, self.dec0 + d_dec)
        
        # 2. Afstand in AU (1 / parallax_rad)
        # Als parallax 0 is (zeer verre ster), gebruik een enorme afstand.
        dist_au = 1.0 / self.parallax if self.parallax > 0 else 1e9
        
        # 3. Positievector in AU t.o.v. SSB
        pos_ssb = u * dist_au
        
        if rate:
            # Snelheidsvector (AU/dag) - optioneel
            # Voor navigatie en aberratie heb je eigenlijk alleen de snelheid 
            # van de aarde nodig, maar we kunnen hem hier schatten:
            vel_ssb = (self._coords_to_vector(self.ra0 + d_ra + self.pm_ra_cosdec/np.cos(self.dec0), 
                                              self.dec0 + d_dec + self.pm_dec) - u) * dist_au
            return pos_ssb, vel_ssb
            
        return pos_ssb
    
    
    
    class CelestialObject:
    def PV_apparent_earth(self, jd, target_id=None, **kwargs):
        """
        Gedeelde pipeline voor zowel Planeten als Sterren.
        """
        clight = self.clight_au_day
        # De 'is_star' check op basis van de klasse-instantie
        is_star = isinstance(self, Star)

        # 1. Toestand van de Aarde (altijd t.o.v. SSB)
        # Deze methode moet de Aarde-positie uit je JPL-data halen
        pv_earth = self.get_earth_pv(jd) 
        pos_earth, vel_earth = pv_earth[0], pv_earth[1]

        # 2. Positie Target (t.o.v. SSB)
        if is_star:
            # STER: Directe berekening op jd (geen jd-tau iteratie)
            pos_t = self.get_ssb_pv(jd)
            pos_geom = pos_t - pos_earth
        else:
            # PLANEET: Iteratieve lichttijd (jd - tau)
            tau = 0.0
            for _ in range(2):
                pos_t = self.get_ssb_pv(jd - tau)
                pos_geom = pos_t - pos_earth
                tau = np.linalg.norm(pos_geom) / clight
        
        dist = np.linalg.norm(pos_geom)
        u = pos_geom / dist

        # 3. Lichtafbuiging door de Zon (GM_Sun)
        # Gebruik de vector-identiteit voor snelheid: q(1+u·q) - u(u·q)
        pos_sun = self.get_sun_pos(jd)
        r_s = pos_sun - pos_earth
        d_s = np.linalg.norm(r_s)
        q = r_s / d_s
        dot_uq = np.dot(u, q)

        if dot_uq > -0.999999: # Voorkom asymmetrie vlak achter de zon
            factor = (4.0 * self.gm_sun) / (clight**2 * d_s * (1.0 + dot_uq))
            u = u + factor * (q * (1.0 + dot_uq) - u * dot_uq)
            u /= np.linalg.norm(u)

        # 4. Aberratie (Eerste orde Lorentz-benadering)
        beta = vel_earth / clight
        u_app = (u + beta) / (1.0 + np.dot(u, beta))
        u_app /= np.linalg.norm(u_app)
        
        return u_app * dist

class Star(CelestialObject):
    def __init__(self, row, reference_jd=2448349.0625):
        super().__init__()
        # ... laad ra0, dec0, pm, parallax uit row ...

    def get_ssb_pv(self, jd):
        # ... jouw lineaire eigenbeweging + parallax logica ...
        return pos_ssb
    

# In de CelestialObject klasse:
pv_earth = self.get_earth_pv(jd)
pos_obs, vel_obs = pv_earth  # Gebruik pos voor parallax, vel voor aberratie

# Lichtafbuiging door de zon:
pos_sun = self.get_sun_pos(jd)
# ... etc ...


