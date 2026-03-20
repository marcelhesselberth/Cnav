#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:28:19 2023

@author: hessel
"""

import os
from re import compile
from time import time
from datetime import datetime
from caltools import JD, MJD, RJD, RMJD
from configparser import ConfigParser
from bisect import bisect_right


path, ext = os.path.splitext(__file__)
config_filename = f"{path}.ini"
config = ConfigParser()
config.read(config_filename)

# Finals
finals_URL = config["Finals"]["URL"]
finals_age = config["Finals"]["maxage"]
finals_dir = config["Finals"]["dir"]
finals_bck = config["Finals"]["backup"]

# Leap seconds
leap_URL    = config["Leap seconds"]["URL"]
leap_age    = config["Leap seconds"]["maxage"]
leap_dir    = config["Leap seconds"]["dir"]
leap_bck    = config["Leap seconds"]["backup"]


class URL_data:
    def __init__(self, url, data_dir, backup_dir, maxage=0, binary=False):
        self.url = url
        self.filename = url.rsplit("/", 1)[-1]
        self.dir = data_dir
        self.backup_dir = backup_dir
        self.path = os.path.join(self.dir, self.filename)
        self.backup_path = os.path.join(self.backup_dir, self.filename)
        self.max_file_age = float(maxage) * 86400  # maxage is given in days
        if binary:
            self.readmode  = "rb"
            self.writemode = "wb"
        else:
            self.readmode  = "r"
            self.writemode = "w"
        self.initialize()


    def initialize(self):
        if not os.path.isdir(self.dir):
            print(f"Creating directory {self.dir}.")
            os.makedirs(self.dir)
        if not os.path.isfile(self.path):
            self.download()
        if os.path.isfile(self.path):
            mtime = os.stat(self.path).st_mtime
            file_age = time() - mtime
            if file_age < 0:
                print(f"File age not valid. Please check system clock.")
            else:
                if file_age > self.max_file_age and self.max_file_age > 0:
                    print(f"{self.filename} is out of date.")
                    self.download()
        try:
            file = open(self.path, self.readmode)
            data = file.read()
            file.close()
        except:
            raise(FileNotFoundError(f"{self.filename}"))
        else:
            self.decode(data)


    def download(self):
        print(f"Downloading {self.url} to {self.path} ...")
        if os.path.isfile(self.path):
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            directory, filename = os.path.split(self.path)
            name, ext = os.path.splitext(filename)
            backup_filename = f"{filename}-{dt}.{ext}"
            backup_path = os.path.join(directory, backup_filename)
            if not os.path.isdir(self.backup_dir):
                print(f"Creating directory {self.backup_dir}.")
                os.makedirs(self.backup_dir)
            os.rename(self.path, backup_path)
            print("Backed up {self.file_path} to {backup_path} .")
        import urllib.request
        urllib.request.urlretrieve(self.url, self.path)


    def decode(self, txt):
        pass


    def __str__(self):
        return self.file_path


class Finals(URL_data):
    def __init__(self, url, path="data", backup_path="backup", maxage=0):
        URL_data.__init__(self, url, path, backup_path, maxage)
        print(self)

    def decode(self, txt):
        self.info = {}
        self.mjd = []
        self.prev_mjd = 0
        self.pred = False
        self.filedate = None
        lines = txt.split('\n')
        for line in lines:
            try:
                yy    = int(line[0:2])
                mm    = int(line[2:4])
                dd    = int(line[4:6])
                mjd   = float(line[7:15])
                pred  = line[57].upper()
                dut1  = float(line[58:68])
                error = float(line[68:78])
            except:
                pass
            else:
                self.add_dut1(yy, mm, dd, mjd, pred, dut1, error)

    def add_dut1(self, yy, mm, dd, mjd, pred, dut1, error):
        if pred == 'P':
            pred = True
        elif pred == 'I':
            pred = False
        else:
            raise(SyntaxError("Entry is neither IERS data nor prediction."))
        if mjd >= 51544:        # Fix date from a 2-digit year
            yyyy = yy + 2000
        else:
            yyyy = yy + 1900
        assert(mjd == MJD(yyyy, mm, dd))
        if mjd > self.prev_mjd:
            self.prev_mjd = mjd
        else:
            raise(ValueError("Dates in IAU 2000 file do not monotonically increase."))
        self.mjd.append(mjd)
        self.info[mjd] = dut1, error
        if pred and not self.pred:  # first prediction, save file date
            self.pred = True
            self.filedate = mjd - 1

    def __str__(self):
        if not self.filedate:
            return ""
        today = datetime.now()
        mjd_today = MJD(today.year, today.month, today.day)
        try:
            y0, m0, d0, t = RMJD(self.filedate)
            dut1, error = self.info[mjd_today]
            ymax, mmax, dmax, t = RMJD(self.mjd[-1])
        except:
            return "Failed to retrieve DUT1 data"
        errstr = "\nDUT1 error today (%d-%0.2d-%0.2d) is %.3f seconds." % \
                (today.year, today.month, today.day, error)
        return "DUT1 data from %d-%0.2d-%0.2d. Data available until %d-%0.2d-%0.2d.%s" % \
                (y0, m0, d0, ymax, mmax, dmax, errstr)

    def __call__(self, mjd):  # DUT1 in seconds, mjd from UTC 
        t = mjd % 1
        mjd = mjd // 1
        try:
            dut1_prev, error = self.info[mjd]
            dut1_next, error = self.info[mjd+1]
        except KeyError:
            print("Warning: DUT1 value not in table. Using 0.")
            return 0
        else:
            if dut1_next - dut1_prev > 0.5:  # next day has positive leap second
                dut1_next -= 1.0
            assert(dut1_next > -1)
            if t > 1.0:
                t = 1.0
            return dut1_prev + (dut1_next - dut1_prev) * t


class Leapseconds(URL_data):
    fmt =  "([0-9]{4})\\s+([A-Z]{3})[\\s]+([0-9]{1,2})[\\s]+=JD\\s([0-9.]+)"
    fmt += "[\\s]+TAI-UTC=[\\s]*([0-9.]+)[\\s]*(S)[\\s]*\\+[\\s]*"
    fmt += "\\([\\s]*MJD[\\s]*-[\\s]*([0-9.]+)[\\s]*\\)[\\s]*X[\\s]*([0-9.]+)"
    fmt += "[\\s]*(S)[\\s]*\n"
    months = { "jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6,   \
              "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12 }

    def __init__(self, url, path="data", backup_path="backup", maxage=0):
        URL_data.__init__(self, url, path, backup_path, maxage)
        print(self)

    def decode(self, txt):
        self.info = {}
        self.jd = []
        self.rinfo = {}
        self.rjd = []
        self.prevjd = 0
        pattern = compile(self.fmt)
        leaps = pattern.findall(txt)
        for leap in leaps:
            y, m, d, jd, offset, s1, mjd, factor, s2 = leap
            self.add_leap(y, m, d, jd, offset, s1, mjd, factor, s2)

    def add_leap(self, y, m, d, jd, offset, s1, mjd, factor, s2):
        if s1.upper() != 'S' or s2.upper() != 'S':
            raise(SyntaxError("unknown unit in leap second file "
                              "(seconds expected)."))
        yyyy = int(y)
        mm   = self.months[m.lower()]
        dd   = int(d)
        jd   = float(jd)
        assert(jd == JD(yyyy, mm, dd))
        offset  = float(offset)
        mjd     = float(mjd)
        factor  = float(factor)
        if jd > self.prevjd:
            self.prevjd = jd
        else:
            raise(ValueError("Dates in leap second file do not "
                             "monotonically increase."))
        self.jd.append(jd)
        self.info[jd] = offset, mjd, factor
        yr, mr, dr, tr = RJD(jd)

        assert(yyyy == yr and mm == mr and dd == dr)
        # checks invariance of reversejd(jd())

    def __str__(self):
        lastleap = self.jd[-1]
        yyyy, mm, dd, t = RJD(lastleap)      # to calculate value
        value = self(yyyy, mm, dd)
        yyyy, mm, dd, t = RJD(lastleap - 1)  # to calculate date
        return "Last leap second was at %d-%0.2d-%0.2d. " \
               "TAI-UTC = %.3f seconds." % (yyyy, mm, dd, value)

    def new_leap_second(self, YYYY, MM, DD):  # day starts with new leap second
        jd = JD(YYYY, MM, DD)
        return (jd in self.jd)

    def __call__(self, YYYY, MM, DD):  # TAI - UTC in seconds, date UTC
        #F = (hh * 3600 + mm * 8 + ss) / 86400
        jd = JD(YYYY, MM, DD)
        mjd = jd - 2400000.5
        last_leap = bisect(self.jd, jd)  # includes today
        if last_leap:
            offset, mjd0, factor = self.info[last_leap]
            return offset + (mjd - mjd0) * factor
        print("Warning: leapsecond value not in table. Using 0.")
        return 0



    
    
class Finals(URL_data):
    def __init__(self, url, path="data", backup_path="backup", maxage=0):
        # Initialiseert de URL_data die vervolgens decode() aanroept
        URL_data.__init__(self, url, path, backup_path, maxage)
        print(self)

    def decode(self, txt):
        self.info = {}
        self.mjd_list = [] # Hernoemd om verwarring met de functie MJD te voorkomen
        self.prev_mjd = 0
        self.pred = False
        self.filedate = None
        self.dXdYvalid = None
        lines = txt.split('\n')
        for line in lines:
            e = 0
            if len(line) < 125: continue
            try:
                mjd = float(line[7:15])
                # We proberen eerst Bulletin B (kolom 135+), anders Bulletin A
                def get_val(start, end, alt_start, alt_end):
                    v = line[start:end].strip()
                    return float(v) if v else float(line[alt_start:alt_end])

                yy, mm, dd = int(line[0:2]), int(line[2:4]), int(line[4:6])
                pred = line[16] # Flag voor Bulletin A (I/P)
                dut1 = get_val(154, 165, 58, 68)
                err_dut1 = float(line[68:78])
                e = 1
                pmx  = get_val(134, 144, 18, 27)
                pmy  = get_val(144, 154, 37, 46)
                e = 2
                dx   = get_val(165, 175, 97, 106)
                dy   = get_val(175, 185, 116, 125)
                e = 3
                self.add_data(yy, mm, dd, mjd, pred, pmx, pmy, dut1, err_dut1, dx, dy)
            except (ValueError, IndexError):
                if e >= 2:
                    self.add_data(yy, mm, dd, mjd, pred, pmx, pmy, dut1, err_dut1, 0, 0)
                    if self.dXdYvalid == None:
                        self.dXdYvalid = mjd
                    


    def add_data(self, yy, mm, dd, mjd, pred, pmx, pmy, dut1, err_dut1, dx, dy):
        is_pred = (pred.upper() == 'P')
        
        # Datum correctie
        yyyy = yy + 2000 if mjd >= 51544 else yy + 1900
        
        # Sla op in dict
        self.info[mjd] = {
            'pmx': pmx, 'pmy': pmy, 'dut1': dut1, 
            'err_dut1': err_dut1, 'dx': dx, 'dy': dy
        }
        
        if mjd > self.prev_mjd:
            self.prev_mjd = mjd
        self.mjd_list.append(mjd)
        
        if is_pred and not self.pred:
            self.pred = True
            self.filedate = mjd - 1

    def __str__(self):
        if not self.info: return "No IERS data loaded."
        
        # Gebruik de laatste 'waargenomen' datum (filedate) of de laatste in de lijst
        ref_mjd = self.filedate if self.filedate else self.mjd_list[-1]
        data = self.info.get(ref_mjd, list(self.info.values())[-1])
        
        y, m, d, _ = RMJD(ref_mjd)
        ymax, mmax, dmax, _ = RMJD(self.mjd_list[-1])
        ydXdY, mdXdY, ddXdY, _ = RMJD(self.dXdYvalid)
        
        return (f"IERS Finals 2000A - Release: {y}-{m:02d}-{d:02d}\n"
                f"DUT1, PMx, PMy valid until: {ymax}-{mmax:02d}-{dmax:02d}\n"
                f"dX, dY valid until: {ydXdY}-{mdXdY:02d}-{ddXdY:02d}\n"
                f"DUT1: {data['dut1']:.7f}s (err: {data['err_dut1']:.4f}s)\n"
                f"PMx: {data['pmx']:.6f}\", PMy: {data['pmy']:.6f}\"\n"
                f"dX: {data['dx']:.3f}ms, dY: {data['dy']:.3f}ms")

    def __call__(self, mjd, full=False):
        t = mjd % 1
        mjd_f = mjd // 1
        try:
            d1, d2 = self.info[mjd_f], self.info[mjd_f + 1]
        except KeyError:
            return 0 if not full else None

        res = {k: d1[k] + t * (d2[k] - d1[k]) for k in ['pmx', 'pmy', 'dx', 'dy', 'err_dut1']}
        
        # DUT1 met leap second fix
        v1, v2 = d1['dut1'], d2['dut1']
        if v2 - v1 < -0.5: v2 += 1.0
        elif v2 - v1 > 0.5: v2 -= 1.0
        res['dut1'] = v1 + t * (v2 - v1)
        return res if full else res['dut1']
        

class Leapseconds(URL_data):
    # Formaat voor USNO tai-utc.dat (bevat MJD kolommen)
    fmt =  "([0-9]{4})\\s+([A-Z]{3})[\\s]+([0-9]{1,2})[\\s]+=JD\\s([0-9.]+)"
    fmt += "[\\s]+TAI-UTC=[\\s]*([0-9.]+)[\\s]*(S)[\\s]*\\+[\\s]*"
    fmt += "\\([\\s]*MJD[\\s]*-[\\s]*([0-9.]+)[\\s]*\\)[\\s]*X[\\s]*([0-9.]+)"
    fmt += "[\\s]*(S)[\\s]*\n"

    months = { "jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6,
               "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12 }

    def __init__(self, url, path="data", backup_path="backup", maxage=0):
        URL_data.__init__(self, url, path, backup_path, maxage)
        # We printen pas na decode
        if self.info:
            print(self)

    def decode(self, txt):
        self.info = {}
        self.mjd_list = []
        self.prev_mjd = 0
        pattern = compile(self.fmt)
        leaps = pattern.findall(txt)
        
        for leap in leaps:
            y, m, d, jd, offset, s1, mjd_ref, factor, s2 = leap
            # Bereken de MJD van de dag waarop de schrikkelseconde ingaat
            mjd_start = float(jd) - 2400000.5
            self.add_leap(y, m, d, mjd_start, offset, mjd_ref, factor)

    def add_leap(self, y, m, d, mjd, offset, mjd_ref, factor):
        if mjd > self.prev_mjd:
            self.prev_mjd = mjd
        else:
            # Sla oude waarden over of gooi error bij foutieve sortering
            return 

        self.mjd_list.append(mjd)
        self.info[mjd] = {
            'offset': float(offset),
            'mjd_ref': float(mjd_ref),
            'factor': float(factor)
        }

    def __str__(self):
        if not self.mjd_list: return "No leap second data."
        last_mjd = self.mjd_list[-1]
        # De waarde op de dag van de laatste sprong
        val = self(last_mjd)
        y, m, d, _ = RMJD(last_mjd)
        return f"Last leap second update: {y}-{m:02d}-{d:02d}. TAI-UTC = {val:.3f}s."

    def new_leap_second(self, mjd):
        """Geeft True als er op deze MJD (om 00:00 UTC) een sprong plaatsvindt."""
        return int(mjd) in self.mjd_list

    def __call__(self, mjd):
        """Geeft TAI - UTC in seconden voor een gegeven MJD."""
        # Zoek de laatste MJD die kleiner of gelijk is aan de gevraagde mjd
        idx = bisect_right(self.mjd_list, mjd)
        
        if idx > 0:
            active_mjd = self.mjd_list[idx - 1]
            data = self.info[active_mjd]
            # Formule: TAI-UTC = BaseOffset + (MJD - MJD_ref) * DriftFactor
            return data['offset'] + (mjd - data['mjd_ref']) * data['factor']
        
        # Voor data vóór 1961 (begin van de tabel)
        return 0
    
    
def finals():
    return Finals(finals_URL, finals_dir, finals_bck, finals_age)

def leapseconds():
    return Leapseconds(leap_URL, leap_dir, leap_bck, leap_age)

if __name__ == "__main__":
    l = leapseconds()
    f = finals()
    print(l(MJD(2026, 3, 17)))
    print(f(MJD(2026, 3, 17), full=True))
    
    