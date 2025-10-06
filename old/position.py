from __future__ import annotations
from datetime import datetime, timedelta
import itertools
import math
import pymap3d as pm
import numpy as np
from numpy.linalg import norm, inv
from numpy import dot, exp
from dataclasses import dataclass
import scipy
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from filterpy.kalman import KalmanFilter

C = 299792458

@dataclass
class Satellite():
    ## CEI Variables
    af0: float # SVClockBias
    af1: float # SVClockDrift
    af2: float # SVClockDriftRate
    Tsv: int # transmission time
    Toe: int
    toc: datetime
    GPSWeek: int
    e: float # Eccentricity
    sqrtA: float
    Cic: float
    Crc: float
    Cis: float
    Crs: float
    Cuc: float
    Cus: float
    DeltaN: float
    Omega0: float
    omega: float
    Io: float
    OmegaDot: float
    IDOT: float
    M0: float
    Tgd: float

    ## Location
    # ECEF
    X: float = None
    Y: float = None
    Z: float = None

    # SV Clock Error
    SVclockErr: float = None
    SVclockRateErr: float = None

    ## Distance
    pseudoRange: float = None

    PRN: str = None # Satellite PRN/ID
    tObs: datetime = None # time of last observation

def tri_lsq(sats):
    def system(p):
        x, y, z, t = p
        eqs = [0] * len(sats)
        i=0
        for sat in sats:
            pr = sat.pseudoRange + C * sat.SVclockErr
            eqs[i] = (C * t) + np.sqrt(((x-sat.X)**2)+((y-sat.Y)**2)+((z-sat.Z)**2)) - pr
            i+=1
        return eqs

    x,y,z,t = leastsq(system, np.asarray((0,0,0,0)))[0]
    return x,y,z,t

def tri_naive(sats):
    # >4 satellites
    if(len(sats) > 3):
        combinations = itertools.combinations(sats, 4)
        count = itertools.combinations(sats, 4)
        comboNum = (sum(1 for _ in count))
        predicted = [[0] * 4] * comboNum
        i = 0
        # for every combination of 4 satellites
        for sat4 in combinations:
            def spheres(p):
                x, y, z, t = p
                return ((C * t) + np.sqrt(((x-sat4[0].X)**2)+((y-sat4[0].Y)**2)+((z-sat4[0].Z)**2)) - sat4[0].pseudoRange,
                        (C * t) + np.sqrt(((x-sat4[1].X)**2)+((y-sat4[1].Y)**2)+((z-sat4[1].Z)**2)) - sat4[1].pseudoRange,
                        (C * t) + np.sqrt(((x-sat4[2].X)**2)+((y-sat4[2].Y)**2)+((z-sat4[2].Z)**2)) - sat4[2].pseudoRange,
                        (C * t) + np.sqrt(((x-sat4[3].X)**2)+((y-sat4[3].Y)**2)+((z-sat4[3].Z)**2)) - sat4[3].pseudoRange)
            # find the intersection of all 4 spheres
            predicted[i] = fsolve(spheres, (0,0,0,0))[0]
            i += 1
        # Statistics
        # Naive prediction, average all combinations
        def naive(predicted):
            i = 1
            [x, y, z, t] = predicted[0]
            while i < comboNum - 1:
                x = (x + predicted[i][0])/2
                y = (y + predicted[i][1])/2
                z = (z + predicted[i][2])/2
                t = (t + predicted[i][3])/2
                i += 1
            return x,y,z,t
        return naive(predicted)

def cei2ecef(sat: Satellite, TRec: datetime):
    ## Constants
    TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time
    GM = 3.986004418e14  # [m^3 s^-2]   Gravitational Constant
    omega_e = 7.2921151467e-5  # [rad s^-1]  Earth Rotation Rate
    F = -4.442807633e-10 # [sec sqrt(m)^-1]  Some Constant IDK
    T0 = TGPS0 + timedelta(weeks = sat.GPSWeek) # Init GPS week

    # Semi-Major Axis
    A = sat.sqrtA ** 2
    # Computed Mean Motion
    n0 = np.sqrt(GM / (A**3)) 
    # Corrected Mean Motion
    n = n0 + sat.DeltaN  

    ## Time Calculations
    tk = sat.Tsv - sat.Toe
    tRec =  (TRec - T0).total_seconds()
    tk = tRec - sat.pseudoRange/C - sat.Toe
    # tk corrections
    if(tk > 302400): tk -= 604800
    elif(tk < -302400): tk+= 604800

    clock_err = sat.af0 + tk * (sat.af1 + tk * sat.af2)
    clock_rate_err = sat.af1 + 2 * tk * sat.af2

    Mk = sat.M0 + n * tk  # Mean Anomaly
    Ek = Mk # Init Eccentric anomaly

    # Ek Caclulations with Corrections
    Ek_old = 2222
    while math.fabs(Ek - Ek_old) > 1.0E-10:
        Ek_old = Ek
        Ek = Ek_old + (Mk - Ek_old + sat.e * np.sin(Ek_old)) / (1.0 - sat.e * np.cos(Ek_old))
    dtr = F * sat.e * sat.sqrtA * np.sin(Ek) # Relativistic correction term
    clock_err += dtr - sat.Tgd

    ## Time Calculations - No Corrections
    #tk = (tsv - toe).total_seconds()
    #Mk = M0 + n * tk  # Mean Anomaly
    #Ek = Mk + e * np.sin(Mk) # Eccentric anomaly

    # True anomaly
    nuK = 2 * np.arctan(np.sqrt((1 + sat.e**2)/(1-sat.e**2)) * np.tan(Ek/2))
    
    # Arguement of Latitude
    PhiK = nuK + sat.omega  
    # argument of latitude correction
    duk = sat.Cuc * np.cos(2 * PhiK) + sat.Cus * np.sin(2 * PhiK)  
    # corrected argument of latitude
    uk = PhiK + duk  
    # inclination correction
    dik = sat.Cic * np.cos(2 * PhiK) + sat.Cis * np.sin(2 * PhiK)  
    # corrected inclination
    ik = sat.Io + sat.IDOT * tk + dik  
    # Radius Correction
    drk = sat.Crc * np.cos(2 * PhiK) + sat.Crs * np.sin(2 * PhiK) 
    # corrected radius
    rk = A * (1 - sat.e * np.cos(Ek)) + drk  
    # Corrected Longitude of ascending node
    OmegaK = sat.Omega0 + (sat.OmegaDot - omega_e) * tk - omega_e * sat.Toe
    ## Position in orbital plane
    Xk1 = rk * np.cos(uk)
    Yk1 = rk * np.sin(uk)

    # ECEF Coordinates
    sat.X = Xk1 * np.cos(OmegaK) - Yk1 * np.sin(OmegaK) * np.cos(ik)
    sat.Y = Xk1 * np.sin(OmegaK) + Yk1 * np.cos(OmegaK) * np.cos(ik)
    sat.Z = Yk1 * np.sin(ik)

    # Clock Err
    sat.SVclockErr = clock_err
    sat.SVclockRateErr = clock_rate_err

def haversine(lon1, lat1, lon2, lat2):
  R = 6371
  dLat = math.radians(lat2-lat1)
  dLon = math.radians(lon2-lon1)
  a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2) 
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  d = R * c
  return d

def iterative(sats, TRec):
    t = 137
    while math.fabs(t) > 1e-10:
        i = 0
        if t == 137 : t = 0
        while i < len(sats):
            sats[i].pseudoRange = sats[i].pseudoRange - C*t
            cei2ecef(sats[i], TRec)
            i += 1
        x,y,z,t = tri_lsq(sats)
    return x,y,z,t

def saast(pos, el, humi=0.75, temp0=15.0):
  """
    Function from RTKlib: https://github.com/tomojitakasu/RTKLIB/blob/master/src/rtkcmn.c#L3362-3362
        with no changes
    :param time:    time
    :param pos:     receiver position {ecef} m)
    :param el:    azimuth/elevation angle {az,el} (rad) -- we do not use az
    :param humi:    relative humidity
    :param temp0:   temperature (Celsius)
    :return:        tropospheric delay (m)
    """
  pos_rad = pm.ecef2geodetic(pos[0],pos[1],pos[2], deg=False)
  if pos_rad[2] < -1E3 or 1E4 < pos_rad[2] or el <= 0:
    return 0.0

  # /* standard atmosphere */
  hgt = 0.0 if pos_rad[2] < 0.0 else pos_rad[2]

  pres = 1013.25 * pow(1.0 - 2.2557E-5 * hgt, 5.2568)
  temp = temp0 - 6.5E-3 * hgt + 273.16
  e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))

  # /* saastamoninen model */
  z = np.pi / 2.0 - el
  trph = 0.0022768 * pres / (
    1.0 - 0.00266 * np.cos(2.0 * pos_rad[0]) - 0.00028 * hgt / 1E3) / np.cos(z)
  trpw = 0.002277 * (1255.0 / temp + 0.05) * e / np.cos(z)
  return trph + trpw
