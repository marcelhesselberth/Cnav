#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:03:57 2026

@author: Marcel Hesselberth
"""


from functools import cache, cached_property, total_ordering
from dataclasses import dataclass
import numpy as np
from webdata import leapseconds, finals
from caltools import JD, MJD, RJD, RMJD, JD2000, MJD0, TJC
from caltools import mdays, wdays, timescales, is_leap_year
from ttmtdb import TTmTDB


TTmTAI = 32.184  # Seconds between TT and TAI time scale
TAImGPS = 19     # Seconds between TAI and GPS time scale
TAImUTC = leapseconds()  # Seconds between the TAI and UTC time scale
new_leap_second = TAImUTC.new_leap_second  # yesterday ended with new leap sec.
UT1mUTC = finals()  # Time difference between UT1 and UTC in seconds


def TTmTDB(tt_jd):  # 10 us precision
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


# =============================================================================
# ATime base class (streamlined)
# =============================================================================


@total_ordering
@dataclass(frozen=True)
class ATime:
    def __init__(self, *args, **kwargs):
        """
        Abstract time class. Instantiate with:
            - A JD
            - A JD tuple (day, fraction)
            - A 6-tuple yyyy, mm, dd, hh, mm, ss+fraction
        The abstract class requires a timescale name as its first argument.
        """
        if len(args) == 2:
            if isinstance(args[1], (tuple, list)) and len(args[1]) == 2:
                # 2-float JD support (int + frac) — high precision
                self.from_jd(*args, **kwargs)
            elif isinstance(args[1], (np.floating, float, int)):
                self.from_jd(*args, **kwargs)
            elif isinstance(args[1], ATime):
                ts, tobj = args
                args = [ts, tobj.year, tobj.month, tobj.day,
                        tobj.hour, tobj.minute, tobj.second]
                self.from_date_tuple(*args, **kwargs)
            else:
                raise TypeError("Wrong time format.")
        elif len(args) == 7:
            self.from_date_tuple(*args, **kwargs)
        else:
            raise TypeError("Wrong time format.")

# =============================================================================
# Helpers for the constructor
# =============================================================================

    def from_jd(self, *args, **kwargs):
        self.tz = 0
        if kwargs != {}:
            raise(ValueError("Got keyword for a Julian date."))
        ts, jd  = args
        if type(ts) != str:
            raise(ValueError("Argument 0 must be a string."))
        ts = ts.upper()
        if not ts in timescales:
            raise(ValueError(f"Unknown time scale ({ts})."))
        if isinstance(jd, (np.floating, float, int)):
            jd1  = jd // 1
            jd2  = jd % 1
        elif isinstance(jd, (tuple, list)) and len(jd) == 2:
            jd1, jd2 = jd
            jd = jd1 + jd2
        else:
            raise(ValueError("Julian date must be float"))
        if jd < 0:
            raise(ValueError("Negative Julian date."))
        self.ts = ts
        self.year, self.month, self.day, T = RJD(jd1)
        if self.year > 9999:
            raise(ValueError("Year must be between -4712 and 9999)."))
        dsec = np.longdouble(jd2) * 86400
        T, fsecond = divmod(dsec, 1)
        T = int(T)
        hh, mm = divmod(T, 3600)
        mm, ss = divmod(mm, 60)
        self.hour = int(hh)
        self.minute = int(mm)
        self.second = float(ss)

    def from_date_tuple(self, *args, **kwargs):
        print("from_tuple", args, kwargs)
        if "offset" in kwargs:
            self.tz = kwargs["tz"]
            print("TZ not implemeted yet")
        elif "TZ" in kwargs:
            self.tz = kwargs["TZ"]
            print("TZ not implemeted yet")
        else: self.tz = 0
        if not isinstance(self.tz, (int, np.integer)):
            raise(ValueError("TZ must be an integer."))
        ts, YYYY, MM, DD, hh, mm, ss = args
        if type(ts) != str:
            raise(ValueError("Argument 0 must be a string."))
        ts = ts.upper()
        if not ts in timescales:
            raise(ValueError(f"Unknown time scale ({ts})."))
        if not isinstance(YYYY, (int, np.integer)):
            raise(ValueError("Year must be an integer."))
        if YYYY < -4712 or YYYY > 9999:
            raise(ValueError("Year must be between -4712 and 9999)."))
        if not isinstance(MM, (int, np.integer)):
            raise(ValueError("Month must be an integer."))
        if MM < 1 or MM > 12:
            raise(ValueError("Month must be between 1 and 12"))
        dmax = mdays[MM]
        if MM == 2 and is_leap_year(YYYY):
            dmax += 1
        if not isinstance(MM, (int, np.integer)):
            raise(ValueError("Month must be an integer."))
        if DD < 1 or DD > dmax:
            raise(ValueError(f"Invalid date ({YYYY}-{MM}-{DD})."))
        if YYYY == 1582 and MM == 10:
            if DD > 4 and DD < 15:  # julian -> gregorian calendar reform
                raise(ValueError(f"Invalid date ({YYYY}-{MM}-{DD})."))
        if not isinstance(hh, (int, np.integer)):
            raise(ValueError("Hour must be an integer."))
        if hh < 0 or hh > 23:
            raise(ValueError("Hour must be between 0 and 23."))
        if not isinstance(mm, (int, np.integer)):
            raise(ValueError("Minute must be an integer."))
        if mm < 0 or mm > 59:
            raise(ValueError("Minute must be between 0 and 60."))
        if ss < 0:
            raise(ValueError("Negative seconds."))
        self.ts = ts
        self.year = YYYY
        self.month = MM
        self.day = DD
        self.hour = hh
        self.minute = mm
        self.second = float(ss)
        if self.has_leap_second():
            if self.second >= 61:
                raise(ValueError("Seconds >= 61 on leap second day."))
        else:
            if self.second >= 60:
                raise(ValueError("Seconds >= 60 on non-leap second day."))

# =============================================================================
# Derived values. The frequently used ones are cached properties.
# =============================================================================

    def dsec(self):
        return 3600 * self.hour + 60 * self.minute + self.second
    
    @cached_property
    def dfrac(self):
        return self.dsec() / 86400
    
    @cached_property
    def jdate(self):
        return JD(self.year, self.month, self.day)

    @cached_property
    def mjdate(self):
        return int(self.jdate - MJD0 + 0.5)

    @cached_property
    def jd(self):
        return self.jdate + self.dfrac
    
    @cached_property
    def jd2(self):
        return self.jdate, self.dfrac

    @cached_property
    def jd2k(self):  # julian date since jd2000 for reduced numerical errors
        return (self.jdate - JD2000) + self.dfrac

    @cached_property
    def tjc(self):
        date = self.jdate
        return TJC(self.jdate, self.dfrac)

    @cached_property
    def mjd(self):
        return self.jd - MJD0

    @cached_property
    def mjd2(self):
        return self.mjdate + self.dsec / 86400
        
    def is_leap_year(self):
        return is_leap_year(self.year)

    def after_leap_second(self):  # day start 00:00:00 just follows leap second
        return new_leap_second(self.mjd)

    def has_leap_second(self):  # day ends with a leap second at 23:59:60
        return new_leap_second(self.mjd)

    def date(self):
        return self.YYYY, self.MM, self.DD, 
    
    def time(self):
        return self.hour, self.minute, self.second

    def datetime(self):
        return self.year, self.month, self.day, \
               self.hour, self.minute, self.second

    def __str__(self):  # iso8601
        YYYY, MM, DD, hh, mm, ss = self.datetime()
        s  = f"{YYYY:04}-{MM:02}-{DD:02}T{hh:02}:{mm:02}:{ss:0>4.6}Z"
        s += f" ({self.ts})"
        return s

    # Nearest neighbor conversions
    def to(self, target_ts: str):
        """Convert to any timescale. A→C works automatically if A→B and B→C exist."""
        target_ts = target_ts.upper()
        if self.ts == target_ts:
            cls = globals()[f"{target_ts}Time"]
            return cls(self.year, self.month, self.day,
                       self.hour, self.minute, self.second)
        # === Direct conversions (only these 6–7 must be maintained) ===
        if self.ts == "UTC" and target_ts == "TAI":   return self.utc2tai()
        if self.ts == "TAI" and target_ts == "UTC":   return self.tai2utc()

        if self.ts == "UTC" and target_ts == "UT1":   return self.utc2ut1()
        if self.ts == "UT1" and target_ts == "UTC":   return self.ut12utc()

        if self.ts == "TAI" and target_ts == "TT":    return TTTime(self + TTmTAI)
        if self.ts == "TT" and target_ts == "TAI":    return TAITime(self - TTmTAI)

        if self.ts == "TAI" and target_ts == "GPS":   return GPSTime(self - TAImGPS)
        if self.ts == "GPS" and target_ts == "TAI":   return TAITime(self + TAImGPS)

        if self.ts == "TT" and target_ts == "TDB":    return self.tt2tdb()
        if self.ts == "TDB" and target_ts == "TT":    return self.tdb2tt()
        
        if self.ts == "TAI" and target_ts == "TDB": return self.tai2tdb()
        if self.ts == "TDB" and target_ts == "TAI": return self.tdb2tai()

        raise ValueError(f"No conversion path from TAI to {target_ts}")


    def ut12utc(self):
        assert self.ts == "UT1"
        dt = UT1mUTC(self.mjd)
        utc = UTCTime(self - dt)
        error = utc.ut1() - self
        while abs(error) > 1e-10:
            utc = utc - error
            error = utc.ut1() - self
        return utc


    def utc2tai(self):
        assert(self.ts == "UTC")
        ls = TAImUTC(self.mjdate)
        tai = TAITime(self) + ls
        return tai
        

    def tai2utc(self):    # used in __sub__ and __add__ if ts=="UTC"
        assert(self.ts == "TAI")
        ls = TAImUTC(self.mjdate)
        utc = UTCTime(self - ls)   # common case
        if self.day == 1:
            if self.after_leap_second():
                dsec = self.dsec()
                if dsec < ls:  # utc date in previous year
                    if dsec >= ls - 1:  # during leap
                        utc.second += 1  # generates 60th second
                    else:
                        utc = UTCTime(self - (ls - 1))  # leap sec in between
        return utc


    def utc2ut1(self):
        assert(self.ts == "UTC")
        dt = UT1mUTC(self.mjd)
        if self.second >= 60 and self.hour == 23 and self.minute == 59:
            if self.has_leap_second():
                tomorrow = self.mjdate + 1  # mjdate discards time
                dt = UT1mUTC(tomorrow) - 1  # value at beginning of next day
        ut1 = UT1Time(self) + dt
        return ut1


    def ut12utc(self):  # TODO: jd
        assert(self.ts == "UT1")
        dt = UT1mUTC(self.mjd)
        utc = UTCTime(self - dt)
        error = utc.ut1() - self
        while error > 1e-10:
            error_old = error
            utc = utc - error
            error = utc.ut1() - self
            assert(error < error_old)            
        return utc


    def tt2tdb(self):
        assert(self.ts == "TT")
        dt = TTmTDB(self.jd)
        tdb = TDBTime(self - dt)
        return tdb


    def tdb2tt(self):
        assert(self.ts == "TDB")
        dt = TTmTDB(self.jd)
        tt = TTTime(self + dt)
        error = tt.tdb() - self
        tt = tt - error
        return tt


    def tai2tdb(self):
        assert self.ts == "TAI"
        tt = self.tt()               # TAI → TT
        tdb = tt.tt2tdb()            # TT → TDB
        return tdb


    def tdb2tai(self):
        assert self.ts == "TDB"
        tt = self.tt()               # TDB → TT
        tai = tt.tai()               # TT → TAI
        return tai


    def __add__(self, x):  # Add x seconds to a time
        if isinstance(x, (np.floating, float, int)):
            if x < 0:
                return self.__sub__(-x)
            #print("add", self, x)
            if self.ts == "UTC":
                t = self.tai()
            else:
                t = self
            d, dsec = divmod(t.dsec() + x, 86400)
            YYYY, MM, DD, T = RJD(self.jdate + d + 0.5)
            #print("d, dsec", d, dsec, YYYY, MM, DD)
            hh, mm = divmod(dsec, 3600)
            mm, ss = divmod(mm, 60)
            hh = int(hh)
            mm = int(mm)
            #print(hh, mm, ss)
            if self.ts == "UTC":
                result = TAITime(YYYY, MM, DD, hh, mm, ss).utc()
            else:
                subclass_type = type(self)
                result = subclass_type(YYYY, MM, DD, hh, mm, ss)
            #print("add result", result)
            return result
        else:
            raise(TypeError("Incompatible operand types for +"))


    def __sub__(self, x):  # subtract x seconds from a time or subtract 2 times
        if isinstance(x, (np.floating, float, int)):  # subtract seconds
            #print("sub", self, x)
            if x < 0:
                return self.__add__(-x)
            if self.ts == "UTC":  # convert to TAI to handle leap seconds
                t = self.tai()
            else:
                t = self
            d, dsec = divmod(t.dsec() - x, 86400)  # no leap seconds here
            YYYY, MM, DD, T = RJD(t.jdate + d + 0.5)
            hh, mm = divmod(dsec, 3600)
            mm, ss = divmod(mm, 60)
            hh = int(hh)
            mm = int(mm)
            if self.ts == "UTC":
                result = TAITime(YYYY, MM, DD, hh, mm, ss).utc()  # convert back
            else:
                subclass_type = type(self)
                result = subclass_type(YYYY, MM, DD, hh, mm, ss)
            #print("sub result",result)
            return result
        elif isinstance(x, ATime):  # difference between 2 time objects
            if self.ts == "UTC" and x.ts == "UTC":  # handle leap seconds
                st = self.tai()
                xt = x.tai()
                days = st.mjdate - xt.mjdate
                seconds = st.dsec() - xt.dsec()
            else:  # Subtracting different time scales is supported.
                days = self.mjdate - x.mjdate
                seconds = self.dsec() - x.dsec()
            return 86400 * days + seconds
        else:
            raise(TypeError("Incompatible operand types for -"))


    def __eq__(self, other):
        if isinstance(other, ATime):
            return self.mjd == other.mjd
        elif isinstance(other, (float, np.float)):
            return self.mjd == other
        else:
            return NotImplementedError


    def __gt__(self, other):
        if isinstance(other, ATime):
            return self.mjd > other.mjd
        elif isinstance(other, (float, np.float)):
            return self.mjd > other
        else:
            return NotImplementedError
        
        
    def __hash__(self):
        return hash((self.year, self.month, self.day, self.hour, self.minute, self.second))


# =============================================================================
# Thin subclasses. Contain the transitive operations.
# =============================================================================

class UTCTime(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("UTC", *args, **kwargs)
    def ut1(self): return self.to("UT1")
    def tai(self): return self.to("TAI")
    def tt(self):  return self.to("TAI").to("TT")
    def gps(self): return self.to("TAI").to("GPS")
    def tdb(self): return self.to("TAI").to("TT").to("TDB")


class TAITime(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("TAI", *args, **kwargs)
    def utc(self): return self.to("UTC")
    def ut1(self): return self.to("UTC").to("UT1")
    def tt(self):  return self.to("TT")
    def gps(self): return self.to("GPS")
    def tdb(self): return self.to("TT").to("TDB")


class UT1Time(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("UT1", *args, **kwargs)
    def utc(self): return self.to("UTC")
    def tai(self): return self.to("UTC").to("TAI")
    def tt(self):  return self.to("UTC").to("TAI").to("TT")
    def gps(self): return self.to("UTC").to("TAI").to("GPS")
    def tdb(self): return self.to("UTC").to("TAI").to("TT").to("TDB")


class TTTime(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("TT", *args, **kwargs)
    def utc(self): return self.to("TAI").to("UTC")
    def ut1(self): return self.to("TAI").to("UTC").to("UT1")
    def tai(self): return self.to("TAI")
    def gps(self): return self.to("GPS")
    def tdb(self): return self.to("TDB")


class GPSTime(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("GPS", *args, **kwargs)
    def utc(self): return self.to("TAI").to("UTC")
    def ut1(self): return self.to("TAI").to("UTC").to("UT1")
    def tai(self): return self.to("TAI")
    def tt(self):  return self.to("TAI").to("TT")
    def tdb(self): return self.to("TAI").to("TT").to("TDB")


class TDBTime(ATime):
    def __init__(self, *args, **kwargs):
        super().__init__("TDB", *args, **kwargs)
    def utc(self): return self.to("TT").to("TAI").to("UTC")
    def ut1(self): return self.to("TT").to("TAI").to("UTC").to("UT1")
    def tai(self): return self.to("TT").to("TAI")
    def tt(self):  return self.to("TT")
    def gps(self): return self.to("TT").to("TAI").to("GPS")


if __name__ == "__main__":
    import sys
    import datetime

    print("=" * 70)
    print("ATime Library Test Suite  —  March 2025 / Leap-second aware")
    print("Current real-world date for reference:", datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    print("=" * 70, "\n")

    # ───────────────────────────────────────────────────────────────
    #  Basic instantiations
    # ───────────────────────────────────────────────────────────────
    print("1. Basic instantiations")
    t_utc_iso = UTCTime(2016, 12, 31, 23, 59, 60.123456)      # leap second example
    t_tai     = TAITime(2025,  1,  1,  0,  0, 31.123456)      # corresponding TAI
    t_ut1     = UT1Time(2024, 12, 31, 23, 59, 59.999)         # near boundary
    t_tt      = TTTime(2025,  1,  1,  0,  1,  0.0)
    t_gps     = GPSTime(2460000.5)             # example JD
    t_tdb     = TDBTime(2025,  1,  1,  0,  0, 32.184)

    print(" UTC:", t_utc_iso)
    print(" TAI:", t_tai)
    print(" UT1:", t_ut1)
    print("  TT:", t_tt)
    print(" GPS:", t_gps)
    print(" TDB:", t_tdb)
    print()

    # ───────────────────────────────────────────────────────────────
    #  Round-trip tests (very important for leap-second days)
    # ───────────────────────────────────────────────────────────────
    print("2. Round-trip consistency (UTC ↔ TAI ↔ TT ↔ TDB ↔ GPS ↔ UT1)")
    tests = [
        UTCTime(2016, 12, 31, 23, 59, 60.0),     # leap second inserted
        UTCTime(2015,  6, 30, 23, 59, 60.0),     # another leap second
        UTCTime(2025, 12, 31, 23, 59, 59.999),   # no leap second (future)
        UTCTime(2024, 10, 10, 12, 34, 56.789),
    ]

    for orig in tests:
        print(f"\nOriginal: {orig}")
        
        tai = orig.tai()
        utc_back = tai.utc()
        print(f"  UTC → TAI → UTC : {'OK' if utc_back == orig else 'FAIL'}   diff={orig - utc_back:.9f} s")

        tt  = orig.tt()
        utc_tt = tt.utc()
        print(f"  UTC → TT → UTC  : {'OK' if abs(utc_tt - orig) < 1e-9 else 'FAIL'}")

        tdb = orig.tdb()
        utc_tdb = tdb.utc()
        print(f"  UTC → TDB → UTC : {'OK' if abs(utc_tdb - orig) < 1e-9 else 'FAIL'}")

        ut1 = orig.ut1()
        utc_ut1 = ut1.utc()
        print(f"  UTC → UT1 → UTC : {'OK' if abs(utc_ut1 - orig) < 1e-8 else 'FAIL'}  (DUT1 tolerance)")

    print()

    # ───────────────────────────────────────────────────────────────
    #  Arithmetic across leap-second boundary
    # ───────────────────────────────────────────────────────────────
    print("3. Arithmetic across leap-second day boundary")
    leap_utc = UTCTime(2016, 12, 31, 23, 59, 59.0)
    print("Start:", leap_utc)

    for dt_sec in [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 10, 86400]:
        result = leap_utc + dt_sec
        print(f"  +{dt_sec:6.1f} s  →  {result}   ({result.ts})")

    print()

    # ───────────────────────────────────────────────────────────────
    #  High-precision 2-float JD instantiation
    # ───────────────────────────────────────────────────────────────
    print("4. High-precision JD input (2-float tuple)")
    jd_int = 2457754
    jd_frac_day = 0.999999999  # almost midnight next day
    t_from_jd = UTCTime((jd_int, jd_frac_day))
    print("From JD:", t_from_jd)
    print("Back to JD:", t_from_jd.jd)
    print("Round-trip error:", abs(t_from_jd.jd - (jd_int + jd_frac_day)))
    print()

    # ───────────────────────────────────────────────────────────────
    #  MJD / TJC / jd2k checks
    # ───────────────────────────────────────────────────────────────
    print("5. MJD, TJC, JD2k consistency")
    ref = UTCTime(2000, 1, 1, 12, 0, 0.0)  # J2000.0 noon
    print("Reference:", ref)
    print(f"  JD   = {ref.jd:.12f}")
    print(f"  JD2k = {ref.jd2k:+.12f}")
    print(f"  MJD  = {ref.mjd:.10f}")
    print(f"  TJC  = {ref.tjc:.10f}")
    print()

    # ───────────────────────────────────────────────────────────────
    #  Quick chain examples
    # ───────────────────────────────────────────────────────────────
    print("6. Chained conversions (should be automatic via TAI hub)")
    sample = UTCTime(2025, 3, 19, 14, 30, 0.0)
    print("UTC sample:", sample)
    print("→ GPS:", sample.gps())
    print("→ TDB:", sample.tdb())
    print("→ UT1:", sample.ut1())
    print("→ back to UTC:", sample.ut1().utc())
    print()

    print("All basic tests completed.")
    print("For production use, also test:")
    print(" • Many more leap-second insertion dates")
    print(" • Sub-microsecond UT1–UTC iterations")
    print(" • Very large time offsets (± centuries)")
    print(" • Negative years / pre-Gregorian dates")
    
    utc1 = UTCTime(2017, 1, 1, 0, 0, 0)
    utc2 = UTCTime(2016, 12, 31, 23, 59, 60.5)
    
    print(utc1-utc2)