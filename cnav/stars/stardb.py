#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:00:22 2026

@author: Marcel Hesselberth
"""

from greek import *
import pandas as pd
import numpy as np
import re
from typing import Optional, List


def load_star_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="string"):
        df[col] = df[col].str.strip()
    # normalize empty strings -> NaN
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    if "HR" in df.columns:
        df["HR"] = pd.to_numeric(df["HR"], errors="coerce").astype("Int64")
    return df


# abbreviation to Greek character
abbr_to_greek = {
    "Alp": "α","Bet": "β","Gam": "γ","Del": "δ","Eps": "ε","Zet": "ζ",
    "Eta": "η","The": "θ","Tet": "θ","Iot": "ι","Kap": "κ","Lam": "λ","Mu": "μ",
    "Nu": "ν","Xi": "ξ","Omi": "ο","Pi": "π","Rho": "ρ","Sig": "σ",
    "Tau": "τ","Ups": "υ","Phi": "φ","Chi": "χ","Psi": "ψ","Ome": "ω"
}


greek_to_abbr = {v: k for k, v in abbr_to_greek.items()}
abbr_to_greek_lc = {k.lower(): v for k, v in abbr_to_greek.items()}


# superscript mappings
superscript_map = {
    "1": "¹", "2": "²", "3": "³",
    "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹"
}
reverse_superscript_map = {v: k for k, v in superscript_map.items()}


# helpers
def to_superscript(num_str):
    return "".join(superscript_map.get(c, c) for c in num_str)

def from_superscript(s):
    return "".join(reverse_superscript_map.get(c, c) for c in s)


# conversion functions between greek letter and abbreviation
def abbr_to_greek_bayer(bayer_str):
    """
    "Alp And"     -> "α And"
    "Alp1 Cen"    -> "α¹ Cen"
    "Bet2 Ori"    -> "β² Ori"
    """
    bayer_str = str(bayer_str)
    parts = bayer_str.strip().split()

    if len(parts) != 2:
        return ""
        raise ValueError(f"Invalid Bayer format: {bayer_str}")

    abbr_part, constellation = parts

    # split possible numeric suffix
    letters = ''.join(filter(str.isalpha, abbr_part))
    digits = ''.join(filter(str.isdigit, abbr_part))
    
    letters = letters.lower()
    greek = abbr_to_greek_lc.get(letters)

    if not greek:
        greek = str(letters)
        #raise ValueError(f"Unknown abbreviation: {letters}")

    if digits:
        greek += to_superscript(digits)

    return f"{greek} {constellation}"

def greek_to_abbr_bayer(bayer_str):
    """
    "α And"   -> "Alp And"
    "α¹ Cen"  -> "Alp1 Cen"
    "β² Ori"  -> "Bet2 Ori"
    """
    parts = bayer_str.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid Bayer format: {bayer_str}")

    greek_part, constellation = parts

    # separate greek letter and superscripts
    base = greek_part[0]
    suffix = greek_part[1:]

    abbr = greek_to_abbr.get(base)
    if not abbr:
        raise ValueError(f"Unknown Greek letter: {base}")

    if suffix:
        digits = from_superscript(suffix)
        abbr += digits

    return f"{abbr} {constellation}"


class Star:
    """
    Represents a single star with high-precision space motion.
    
    Position and velocity are computed using rigorous linear 3D propagation
    (straight-line motion through space), which is the most accurate simple model
    for stellar proper motion over centuries.
    """

    def __init__(self, row: pd.Series, reference_jd: float = 2451545.0):
        """
        row: pandas Series containing one star's data from the DataFrame
        reference_jd: Julian Date of the catalog epoch (default = J2000.0)
        """
        self.proper_name = str(row["ProperName"]).strip()
        self.bayer_abbr = str(row["BayerName"]).strip()
        self.cst = str(row["Cst"]).strip()
        
        # Full Bayer designations
        self.full_bayer_abbr = f"{self.bayer_abbr} {self.cst}"
        try:
            self.full_bayer_greek = abbr_to_greek_bayer(self.full_bayer_abbr)
        except Exception:
            self.full_bayer_greek = self.full_bayer_abbr

        self.vmag = float(row["Vmag"])
        self.ra0 = float(row["ra_rad"])
        self.dec0 = float(row["dec_rad"])
        self.pi_rad = float(row["pi_rad"])
        self.mu_alpha_cosdec = float(row["mu_alpha_cosdec_rad_d"])
        self.mu_delta = float(row["mu_delta_rad_d"])
        
        self.hr = int(row["HR"]) if pd.notna(row.get("HR")) else None
        self.hip = int(row["HIP"]) if pd.notna(row.get("HIP")) else None
        
        self.reference_jd = float(reference_jd)

    def get_pv(self, jd: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Most precise version:
        Returns (position, velocity) at the given Julian Date.
        
        position : np.ndarray (3,)  - barycentric ICRS cartesian position in AU
        velocity : np.ndarray (3,)  - barycentric cartesian velocity in AU/day (constant)
        
        Uses linear space motion: r(t) = r0 + v * dt
        Radial velocity is assumed to be zero (standard when not provided).
        """
        # Reference direction unit vector at catalog epoch
        alpha = self.ra0
        delta = self.dec0

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cd = np.cos(delta)
        sd = np.sin(delta)

        u0 = np.array([cd * ca, cd * sa, sd])          # unit vector

        # Distance in AU from parallax (in radians)
        if abs(self.pi_rad) < 1e-12:                   # extremely distant / no parallax
            return np.full(3, np.nan), np.zeros(3)

        d0_au = 1.0 / self.pi_rad

        # Reference position vector at catalog epoch
        r0 = d0_au * u0

        # Tangential velocity from proper motion
        # Orthogonal unit vectors in the tangent plane
        unit_east = np.array([-sa, ca, 0.0])                    # increasing RA
        unit_north = np.array([-sd * ca, -sd * sa, cd])         # increasing Dec

        mu_tangent = (self.mu_alpha_cosdec * unit_east +
                      self.mu_delta * unit_north)

        v = d0_au * mu_tangent          # velocity in AU/day

        # Linear propagation
        dt_days = jd - self.reference_jd
        pos = r0 + v * dt_days

        return pos, v

    def get_position(self, jd: float) -> np.ndarray:
        """Convenience method: returns only the position vector (AU)."""
        pos, _ = self.get_pv(jd)
        return pos

    def get_distance_au(self, jd: float) -> float:
        """Returns distance from barycenter in AU at given date."""
        pos = self.get_position(jd)
        return np.linalg.norm(pos)

    def __repr__(self):
        name = self.proper_name if self.proper_name != "nan" else self.full_bayer_greek
        return f"Star({name}, V={self.vmag:.2f}, dist≈{self.get_distance_au(self.reference_jd):.0f} AU)"

    def __str__(self):
        return (f"{self.proper_name} / {self.full_bayer_greek}\n"
                f"   Vmag = {self.vmag:.2f}\n"
                f"   RA/Dec (J2000) = {np.degrees(self.ra0):.4f}° / {np.degrees(self.dec0):.4f}°\n"
                f"   Parallax = {self.pi_rad*206265000:.2f} μas")    


class StarDB:
    """
    Star database with case-insensitive wildcard search (? and *).
    Properly handles cases where Bayer names like "Alp Cen" point to ProperName "Rigil Kentaurus".
    """

    def __init__(self, filename: str = "stars.csv", reference_jd: float = 2451545.0):
        self.df = load_star_csv(filename)

        # Clean columns
        self.df["ProperName"] = self.df["ProperName"].astype(str).str.strip()
        self.df["BayerName"] = self.df["BayerName"].astype(str).str.strip()
        self.df["Cst"] = self.df["Cst"].astype(str).str.strip()
        
        self.df["full_bayer_abbr"] = (self.df["BayerName"] + " " + self.df["Cst"]).str.strip()

        def safe_greek(b_str: str) -> str:
            try:
                return abbr_to_greek_bayer(b_str)
            except Exception:
                return pd.NA

        self.df["full_bayer_greek"] = self.df["full_bayer_abbr"].apply(safe_greek)

        self.reference_jd = reference_jd

    def _term_to_regex_tokens(self, term: str) -> List[str]:
        """Convert user wildcards to regex. ? = exactly 1 char, * = 0+ chars."""
        tokens = re.split(r'\s+', term.strip())
        regex_tokens = []
        for t in tokens:
            if not t:
                continue
            escaped = re.escape(t)
            regex = escaped.replace(r'\?', '.').replace(r'\*', '.*')
            regex_tokens.append(regex)
        return regex_tokens

    def _term_to_regex_tokens(self, term: str) -> List[str]:
        tokens = re.split(r'\s+', term.strip())
        regex_tokens = []
        for t in tokens:
            if not t: continue
            
            # Escape text (Sirius remains Sirius, Alp? wordt Alp\?)
            escaped = re.escape(t)
            
            # Replace \? by . (1 char) followed by \b (end of word)
            # This makes Alp? match met Alp1, but not Alpha
            # Replace \* door .* (0 or more characters)
            pattern = escaped.replace(r'\?', r'.\b').replace(r'\*', r'.*')
            
            regex_tokens.append(rf"\b{pattern}")
            
        return regex_tokens
        
    def _search_dataframe(self, term: str) -> pd.DataFrame:
        if not term or not str(term).strip():
            return pd.DataFrame()

        regex_tokens = self._term_to_regex_tokens(term)

        # === 1. Try ProperName first ===
        mask = pd.Series(True, index=self.df.index)
        for regex in regex_tokens:
            mask &= self.df["ProperName"].str.contains(regex, case=False, regex=True, na=False)
        results = self.df[mask]

        # === 2. If no results, search Bayer (this is key for "alp? cen*") ===
        if len(results) == 0:
            mask = pd.Series(True, index=self.df.index)
            for regex in regex_tokens:
                abbr_mask = self.df["full_bayer_abbr"].str.contains(
                    regex, case=False, regex=True, na=False
                )
                greek_mask = self.df["full_bayer_greek"].str.contains(
                    regex, case=False, regex=True, na=False
                )
                mask &= (abbr_mask | greek_mask)
            results = self.df[mask]

        # Sort by name length for more "natural" first result
        if not results.empty:
            results = results.copy()
            results["_sort_len"] = results["ProperName"].str.len()
            results = results.sort_values("_sort_len").drop(columns=["_sort_len"])

        return results

    def _find_row(self, ident: str) -> pd.Series | None:
        ident = str(ident).strip()
        if not ident:
            return None
        results = self._search_dataframe(ident)
        return results.iloc[0].copy() if not results.empty else None

    def search(self, term: str, limit: int = 50) -> List['Star']:
        """Main search: returns list of Star objects."""
        results_df = self._search_dataframe(term)
        if len(results_df) > limit:
            results_df = results_df.head(limit)

        return [Star(row, reference_jd=self.reference_jd) 
                for _, row in results_df.iterrows()]

    def get_star(self, identifier: str) -> Optional['Star']:
        row = self._find_row(identifier)
        return Star(row, reference_jd=self.reference_jd) if row is not None else None

    def get_stars(self, identifiers: str | list[str]) -> list['Star']:
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        return [s for ident in identifiers if (s := self.get_star(ident)) is not None]

    def get_positions(self, identifiers: str | list[str], jd: float) -> np.ndarray:
        stars = self.get_stars(identifiers)
        if not stars:
            return np.empty((0, 3))
        return np.array([s.get_position(jd) for s in stars])

    def get_pvs(self, identifiers: str | list[str], jd: float) -> tuple[np.ndarray, np.ndarray]:
        stars = self.get_stars(identifiers)
        if not stars:
            return np.empty((0, 3)), np.empty((0, 3))
        pvs = [s.get_pv(jd) for s in stars]
        pos = np.array([p[0] for p in pvs])
        vel = np.array([p[1] for p in pvs])
        return pos, vel

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"StarDB({len(self.df)} stars, epoch JD={self.reference_jd})"
    
    
if __name__ == "__main__":
    db = StarDB("stars.csv")
    sirius = db.get_star("sirius")
    print(sirius)                                # Star(Sirius, α CMa)
    
    # Single star position at a specific date
    pos = sirius.get_position(2460000.0)         # numpy array (3,)
    print(pos)
    
    # Batch for your pipeline (returns (N,3) array ready for numpy processing)
    positions = db.get_positions(["Sirius", "α CMa", "Alp Boo", "Vega"], jd=2460000.0)
    # positions.shape == (4, 3)
    print(positions)
    
    # Or get PV for light-time correction
    pos, vel = db.get_pvs(["Sirius", "Canopus"], jd=2460000.0)
    
    print(db.search("rigil kent*"))

    print("alp? cen* →", [s.proper_name for s in db.search("alp? cen*")])
    print("rigil kent* →",   [s.proper_name for s in db.search("rigil kent*")])
    print("al? cen* →",      [s.proper_name for s in db.search("al? cen*")])
    print("vega →",          [s.proper_name for s in db.search("vega")])