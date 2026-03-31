#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:14:35 2026

@author: Marcel Hesselberth
"""

import pandas as pd
import numpy as np
from astroquery.vizier import Vizier

# --- CONFIG ---
CATALOG_YALE = "V/50/catalog"
CATALOG_HIP_CROSS = "I/239/hip_main"
CATALOG_HIP2 = "I/311/hip2"
IAU_FILE = 'IAU-Catalog.csv'
TEST_LIMIT = -1  # Zet op -1 voor de volledige catalogus

# Constellation mapping
cst_full = {
    'And': 'Andromeda', 'Ant': 'Antlia', 'Aps': 'Apus', 'Aqr': 'Aquarius', 'Aql': 'Aquila',
    'Ara': 'Ara', 'Ari': 'Aries', 'Aur': 'Auriga', 'Boo': 'Bootes', 'Cae': 'Caelum',
    'Cam': 'Camelopardalis', 'Cnc': 'Cancer', 'CVn': 'Canes Venatici', 'CMa': 'Canis Major',
    'CMi': 'Canis Minor', 'Cap': 'Capricornus', 'Car': 'Carina', 'Cas': 'Cassiopeia',
    'Cen': 'Centaurus', 'Cep': 'Cepheus', 'Cet': 'Cetus', 'Cha': 'Chamaeleon',
    'Cir': 'Circinus', 'Col': 'Columba', 'Com': 'Coma Berenices', 'CrA': 'Corona Australis',
    'CrB': 'Corona Borealis', 'Crv': 'Corvus', 'Crt': 'Crater', 'Cru': 'Crux',
    'Cyg': 'Cygnus', 'Del': 'Delphinus', 'Dor': 'Dorado', 'Dra': 'Draco', 'Equ': 'Equuleus',
    'Eri': 'Eridanus', 'For': 'Forax', 'Gem': 'Gemini', 'Gru': 'Grus', 'Her': 'Hercules',
    'Hor': 'Horologium', 'Hya': 'Hydra', 'Hyi': 'Hydrus', 'Ind': 'Indus', 'Lac': 'Lacerta',
    'Leo': 'Leo', 'LMi': 'Leo Minor', 'Lep': 'Lepus', 'Lib': 'Libra', 'Lup': 'Lupus',
    'Lyn': 'Lynx', 'Lyr': 'Lyra', 'Men': 'Mensa', 'Mic': 'Microscopium', 'Mon': 'Monoceros',
    'Mus': 'Musca', 'Nor': 'Norma', 'Oct': 'Octans', 'Oph': 'Ophiuchus', 'Ori': 'Orion',
    'Pav': 'Pavo', 'Peg': 'Pegasus', 'Per': 'Perseus', 'Phe': 'Phoenix', 'Pic': 'Pictor',
    'Psc': 'Pisces', 'PsA': 'Piscis Austrinus', 'Pup': 'Puppis', 'Pyx': 'Pyxis',
    'Ret': 'Reticulum', 'Sge': 'Sagitta', 'Sgr': 'Sagittarius', 'Sco': 'Scorpius',
    'Scl': 'Sculptor', 'Sct': 'Scutum', 'Ser': 'Serpens', 'Sex': 'Sextans', 'Tau': 'Taurus',
    'Tel': 'Telescopium', 'Tri': 'Triangulum', 'TrA': 'Triangulum Australe', 'Tuc': 'Tucana',
    'UMa': 'Ursa Major', 'UMi': 'Ursa Minor', 'Vel': 'Vela', 'Vir': 'Virgo', 'Vol': 'Volans',
    'Vul': 'Vulpecula'
}

# 1. Build IAU dict (Proper Names)
try:
    iau_df = pd.read_csv(IAU_FILE)
    # Remove rows without name, duplicate ID's
    iau_df = iau_df.dropna(subset=['ProperName'])
    
    # HR Mapping: Extract number from "HR 998"
    iau_hr_map = iau_df[iau_df['Designation'].str.contains('HR', na=False)].copy()
    iau_hr_map['HR_num'] = iau_hr_map['Designation'].str.extract('(\d+)').astype(float)
    dict_hr = iau_hr_map.drop_duplicates('HR_num').set_index('HR_num')['ProperName'].to_dict()
    
    # HIP Mapping: Directly via HIP col.
    iau_hip_map = iau_df.dropna(subset=['HIP']).copy()
    iau_hip_map['HIP_num'] = iau_hip_map['HIP'].astype(float)
    dict_hip = iau_hip_map.drop_duplicates('HIP_num').set_index('HIP_num')['ProperName'].to_dict()
    print("IAU Dictionary succesvol geladen.")
except FileNotFoundError:
    print(f"Let op: {IAU_FILE} niet gevonden. ProperNames blijven leeg.")
    dict_hr, dict_hip = {}, {}

# 2. Fetch data via Vizier (** to get all columns)
v = Vizier(columns=['**'], row_limit=TEST_LIMIT)
yale = v.query_constraints(catalog=CATALOG_YALE, Vmag="<6.5")[0].to_pandas()
cross = v.query_constraints(catalog=CATALOG_HIP_CROSS, HD=">0")[0].to_pandas()
hip2 = v.query_constraints(catalog=CATALOG_HIP2)[0].to_pandas()

# 3. Merge (Yale -> Cross-index -> HIP2)
df_mid = pd.merge(yale, cross[['HD', 'HIP']], on='HD', how='inner')
df = pd.merge(df_mid, hip2, on='HIP', how='inner', suffixes=('_yale', ''))

# 4. Convert to radians (use HIP2 data voor high precision)
deg2rad = np.pi / 180.0
mas2rad = deg2rad / 3600000.0
mas2rad_day = mas2rad / 365.25

df['ra_rad'] = df['RArad'] * deg2rad
df['dec_rad'] = df['DErad'] * deg2rad
df['pi_rad'] = df['Plx'] * mas2rad
df['mu_alpha_cosdec_rad_d'] = df['pmRA'] * mas2rad_day
df['mu_delta_rad_d'] = df['pmDE'] * mas2rad_day

# 5. Map names and constellations
df['Cst'] = df['Name'].str.strip().str[-3:]
df['Constellation_Full'] = df['Cst'].map(cst_full).fillna(df['Cst'])

# BayerName ('alp', 'bet', 'alp1')
if 'Bayer' in df.columns:
    df['BayerName'] = df['Bayer'].str.strip().fillna("")
else:
    # Fallback: Yale Name string
    df['BayerName'] = df['Name'].str[0:-3].str.strip().fillna("")
    
# More sophisticated extraction
# Uses the 10 char Name minus the 3 char cst
ident = df['Name'].str[:-4].str.strip()

# ^\d+       find digit at start
# (?=[a-zA-Z]) but only if followed by a letter (otherwise flamsteed nr)
df['BayerName'] = ident.str.replace(r'^\d+(?=[a-zA-Z])', '', regex=True).str.strip()

# ProperName mapping (IAU)
df['ProperName'] = df['HR'].astype(float).map(dict_hr)
hip_fallback = df['HIP'].astype(float).map(dict_hip)
df['ProperName'] = df['ProperName'].fillna(hip_fallback).fillna("")

# If ProperName is empty expansion is possible using the constellation
# df.loc[df['ProperName'] == "", 'ProperName'] = df['BayerName'] + " " + df['Cst']

# ProperName mapping: First HR, then HIP, otherwise empty string
df['ProperName'] = df['HR'].astype(float).map(dict_hr)
hip_fallback = df['HIP'].astype(float).map(dict_hip)
df['ProperName'] = df['ProperName'].fillna(hip_fallback).fillna("")

# Clean up HR
df['HR'] = pd.to_numeric(df['HR'], errors='coerce').fillna(-1).astype(int)

# 6. Select data
final_cols = [
    'ProperName', 'BayerName', 'Cst', 'Constellation_Full', 'Vmag', 
    'ra_rad', 'dec_rad', 'pi_rad', 
    'mu_alpha_cosdec_rad_d', 'mu_delta_rad_d', 'HR', 'HIP'
]

stars_final = df.drop_duplicates(subset=['HIP']).sort_values('Vmag')

# 7. Save
stars_final[final_cols].to_csv('stars.csv', index=False)

print(f"Ready! Processed {len(stars_final)} stars.")
print(stars_final[['ProperName', 'BayerName', 'Cst', 'Vmag']].head(15))
