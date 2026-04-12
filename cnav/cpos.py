#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:14:03 2026

@author: Marcel Hesselberth
"""

import datetime

class Cpos:
    """
    Basisklasse voor Celestial Positions. 
    Fungeert als Factory: geeft Ppos (Planeten) of Spos (Sterren) terug.
    """
    PLANETEN = ['zon', 'maan', 'mercurius', 'venus', 'mars', 'jupiter', 'saturnus', 'uranus', 'neptunus']

    def __new__(cls, body, ctime):
        # 1. Selecteer de juiste subklasse op basis van de body-naam
        if body.lower() in cls.PLANETEN:
            target_class = Ppos
        else:
            target_class = Spos
        
        # 2. Maak de instantie aan van de subklasse
        instance = object.__new__(target_class)
        return instance

    def __init__(self, body, ctime):
        self.body = body
        self.ctime = ctime # ctime object met tijdschaal conversies
        self.vector_gcrs = None # De uiteindelijke vector na pijplijn

    def transform_to_local(self):
        """
        Gedeelde logica voor alle hemellichamen ná de specifieke pijplijn.
        """
        # A. Lichtafbuiging (Gravitational Deflection)
        # Gebruik Inpop(Sun, SSB, tijd) om hoek met de zon te bepalen.
        self._apply_gravitational_deflection()

        # B. Stellaire Aberratie (Speciale Relativiteit)
        # Gebruik de snelheidsvector van de aarde uit Inpop.
        self._apply_stellar_aberration()

        # C. GCRS naar TIRS (IAU2006)
        # Precessie, Nutatie, ERA (Aardrotatie).
        self._apply_iau2006_transform()

        # D. Topocentrisch (Optioneel)
        # Diurnale aberratie en parallax op basis van lokatie waarnemer.
        pass

    def _apply_gravitational_deflection(self):
        # Implementatie voor zwaartekrachtbuiging zon/planeten
        pass

    def _apply_stellar_aberration(self):
        # Jouw code op basis van speciale relativiteit (Lorentz boost)
        pass

    def _apply_iau2006_transform(self):
        # Transformatiematrix voor aardrotatie en as-oriëntatie
        pass


class Ppos(Cpos):
    """Pijplijn voor lichamen binnen het zonnestelsel."""
    def __init__(self, body, ctime):
        super().__init__(body, ctime)
        self.execute_planetary_pipeline()

    def execute_planetary_pipeline(self):
        # 1. Haal Aarde/SSB positie op via Inpop
        # 2. Iteratieve Lichttijdcorrectie:
        #    - Bereken afstand, bepaal vertrektijd licht (t - tau)
        #    - Inclusief Shapiro delay in de iteratie
        # 3. Bereken relatieve vector: Positie(Body @ t-tau) - Positie(Aarde @ t)
        self.vector_gcrs = "Resultaat van Inpop iteratie"


class Spos(Cpos):
    """Pijplijn voor sterren (buiten het zonnestelsel)."""
    def __init__(self, body, ctime):
        super().__init__(body, ctime)
        self.execute_stellar_pipeline()

    def execute_stellar_pipeline(self):
        # 1. Haal Hipparcos data op (Star klasse)
        # 2. Pas Proper Motion toe tot aan ctime
        # 3. Pas Jaarlijkse Parallax toe (Aardpositie vs SSB nodig via Inpop)
        # 4. Resultaat is de geometrische vector (zonder lichttijd iteratie)
        self.vector_gcrs = "Resultaat van Star + Parallax"

# --- Gebruik ---
# pos = Cpos("Mars", mijn_ctime_object)
# pos.transform_to_local()
