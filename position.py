
from __future__ import annotations
import copy
from datetime import datetime, timedelta
import math
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from scipy.optimize import fsolve
import pymap3d as pm

# Constants
C = 299792458 # Speed of light
omega_e = 7.2921151467e-5 # Earth's rotation rate
TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time

# Satellite Object as Dataclass
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

    # Satellite PRN/ID
    PRN: str = None 

    ## Location
    # ECEF
    X: float = None
    Y: float = None
    Z: float = None

    # SV Clock Error
    SVclockErr: float = None
    SVclockRateErr: float = None

    # PseudoRange
    pseudoRange: float = None

    # Observation
    tObs: datetime = None # time of last observation

import parse

'''
Convert from ECEF to ellipsoidal elements
Parameters:
    x, y, z: ECEF coordinates
Output: 
    lambda, phi, h: ellipsoidal elements
Method: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
'''
def ecef2elli(x,y,z):
    a = 6378137
    f = 1/298.257
    e = math.sqrt(2*f-f**2)
    lamb = math.atan2(y,x)
    p = math.sqrt(x**2+y**2)
    h = 0
    phi = math.atan2(z, p*(1-e**2))
    N = a/(1-(e*math.sin(phi))**2)**0.5 
    delta_h = 1000000
    while delta_h > 0.01:
        prev_h = h
        phi = math.atan2(z, p*(1-e**2*(N/(N+h))))
        N = a/(1-(e*math.sin(phi))**2)**0.5
        h = p/math.cos(phi)-N
        delta_h = abs(h-prev_h)
    return lamb, phi, h

'''
Calculate User Position
Parameters:
    xs: Satellite position matrix
    pr: corrected pseudoranges
    numSat: number of satellites
    x0: starting estimate of user position
    b0: starting estimate of user clock bias
Output: 
    x: optimized user position
    b: optimized user clock bias
    norm_dp: normalized pseudo-range difference
    G: user satellite geometry matrix
Method: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
'''
def estimate_position(xs, pr, numSat, x0, b0):
    dx = [100,100,100]
    db = 0
    norm_dp = 100
    numIter = 0
    b = b0
    #while (norm_dp > 1e-4):
    while norm(dx) > 1e-3:
        diff = [np.subtract(i, x0) for i in xs]
        pow = np.power(diff,2)
        norms = np.sqrt([sum(i) for i in pow]) 
        dp = np.subtract(pr, norms)
        dp = np.add(dp, b)
        dp = np.subtract(dp, b0)
        G = np.multiply(diff, -1)
        for i in range(len(norms)):
            G[i] = np.divide(G[i], norms[i])
        G = [np.append(row,1) for row in G]
        sol = (np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G)) @ dp
        
        dx = np.transpose([sol[0],sol[1],sol[2]])
        db = sol[3]

        norm_dp = norm(dp)
        numIter = numIter + 1
        x0 = x0 + dx
        b0 = b0 + db

    return x0, b0, norm_dp, G

'''
Calculate Satellite Position
Parameters:
    eph: satellite object
    t: transmission time (relative to GPSWeek)
Output: 
    x, y, z: satellite ECEF coordinates
Method: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
'''
def sat_position(eph, t):
    mu = 3.986005e14
    omega_dot_earth = 7.2921151467e-5
    A = eph.sqrtA ** 2
    cmm = math.sqrt(mu/A**3)
    tk = t - eph.Toe
    if (tk > 302400):
        tk = tk-604800
    if (tk < -302400):
        tk = tk+604800
    n = cmm + eph.DeltaN
    mk = eph.M0 + n*tk
    def f(E):
        return E - eph.e*math.sin(E) - mk
    Ek = fsolve(f, mk)

    nu = math.atan2((math.sqrt(1-eph.e**2))*math.sin(Ek)/(1-eph.e*math.cos(Ek)), (math.cos(Ek)-eph.e)/(1-eph.e*math.cos(Ek)))
    #Ek = math.acos((eph.e  + math.cos(nu))/(1+eph.e*math.cos(nu)))
 
    Phi = nu + eph.omega
    du = eph.Cus*math.sin(2*Phi) + eph.Cuc*math.cos(2*Phi)
    dr = eph.Crs*math.sin(2*Phi) + eph.Crc*math.cos(2*Phi)
    di = eph.Cis*math.sin(2*Phi) + eph.Cic*math.cos(2*Phi)
    u = Phi + du
    r = A*(1-eph.e*math.cos(Ek)) + dr
 
    i = eph.Io + eph.IDOT*tk + di
    x_prime = r*math.cos(u)
    y_prime = r*math.sin(u)
    omega = eph.Omega0 + (eph.OmegaDot - omega_dot_earth)*tk - omega_dot_earth*eph.Toe
 
    eph.X = x = x_prime*math.cos(omega) - y_prime*math.cos(i)*math.sin(omega)
    eph.Y = y = x_prime*math.sin(omega) + y_prime*math.cos(i)*math.cos(omega)
    eph.Z = z = y_prime*math.sin(i)

    return x,y,z

'''
Calculate Satellite Bias
Parameters:
    sat: satellite object
    t: reciever time (relative to GPSWeek)
Output: 
    dsv: satellite clock bias
Method: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
'''
def satbias(sat, t):
    F = -4.442807633e-10
    mu = 3.986005e14
    A = sat.sqrtA ** 2
    cmm = math.sqrt(mu/A**3)
    tk = t - sat.Toe
    if (tk > 302400):
        tk = tk-604800
    if (tk < -302400):
        tk = tk+604800
    n = cmm + sat.DeltaN
    mk = sat.M0 + n*tk
    T0 = TGPS0 + timedelta(weeks = sat.GPSWeek) # Init GPS week
    Toc = (sat.toc - T0).total_seconds()
    def f(E):
        return E - sat.e*math.sin(E) - mk
    Ek = fsolve(f, mk, xtol = 0.1)
    dsv = sat.af0 + sat.af1 * (t-Toc) + sat.af2 * (t-Toc)**2 + F * sat.e * sat.sqrtA * math.sin(Ek)
    return dsv

'''
Calculate Distance between 2 lat/lon points using Haversine Method
Parameters:
    lon1: lon of location 1
    lat1: lat of location 1
    lon2: lon of location 2
    lat2: lat of location 2
Output: 
    Distance (kilometers)
Method: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
'''
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # this is in miles.  For Earth radius in kilometers use 6372.8 km

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*math.asin(np.sqrt(a))

    return R * c

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
def saast(pos, el, humi=0.75, temp0=15.0):

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

def iono(sat, alpha, beta, GPSTime, xu):
    LambdaU, PhiU, h = ecef2elli(xu[0],xu[1],xu[2])
    lat = PhiU*180/np.pi
    lng = LambdaU*180/np.pi
    enu = pm.ecef2enu(sat.X, sat.Y, sat.Z, lat, lng, h)
    A = np.arctan2(enu[0], enu[1])
    E = np.arcsin(enu[2]/norm(enu))
    psi = (0.0137/(E + 0.11)) - 0.022
    F = 1.0 + 16.0*((0.53-E)**3)
    PhiI = PhiU + psi * np.cos(A)
    if PhiI > 0.416: PhiI = 0.416
    if PhiI < -0.416: PhiI = -0.416
    LambdaI = LambdaU + ((psi*np.sin(A))/(np.cos(PhiI)))
    PhiM = PhiI + 0.064*np.cos(LambdaI - 1.617)
    T0 = TGPS0 + timedelta(weeks = sat.GPSWeek) # Init GPS week
    GPST = (GPSTime- T0).total_seconds()
    t = 4.32*(10**4)*LambdaI + GPST
    if t < 0: t += 86400
    if t >= 86400: t += 86400
    PER = 0
    for i in range(3):
        PER += beta[i] * (PhiM ** i)
    if PER < 72000: PER = 72000
    x = (2 * np.pi * (t - 50400))/PER
    AMP = 0
    for i in range(3):
        AMP += alpha[i] * (PhiM ** i)
    if AMP < 0: AMP = 0
    Tion = 0
    if np.fabs(x) < 1.57: Tion = F * (5 * (10**-9) + AMP * (1 - ((x**2)/2) + ((x**4)/24)))
    if np.fabs(x) >= 1.57: Tion = F * (5*(10**-9))
    return Tion

'''
Process RINEX files
Parameters:
    nav: navigation file path
    obs: observation file path
    pos: position file path (optional)
Spoofing Parameters:
    pr_shifts: constant to add to pseudorange for each satellite of interest {PRN: [initial, increment]} (optional) 
Output: 
    result = {
        obsTime: {
            "X", "Y", "Z": ECEF coordinates of receiver
            "numSV": number of SVs at observation
            "clockBias": receiver clock bias at observation
            "err": error from true position (= None if no position file provided)
        }
    }
Method: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
'''
def position(nav: str, obs: str, pos: str = None, pr_shifts = {}):
    """
    #Plotting Stuff
    iteration_valid = 0
    y1points = {'G18':[], 'G23':[], 'G05':[], 'G24':[],  'G13':[],  'G15':[],  'G29':[],  'G10':[]}
    y2points = {'G18':[], 'G23':[], 'G05':[], 'G24':[],  'G13':[],  'G15':[],  'G29':[],  'G10':[]}
    
    fig, axs = plt.subplots(4, 2, sharex = True)
    fig.suptitle('Difference in Pseudorange over Time') 
    axs[0, 0].set_title('G18')
    axs[0, 1].set_title('G23')
    axs[1, 0].set_title('G05')
    axs[1, 1].set_title('G24')
    axs[2, 0].set_title('G13')
    axs[2, 1].set_title('G15')
    axs[3, 0].set_title('G29')
    axs[3, 1].set_title('G10')
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Difference in PseudoRange (m)', va='center', rotation='vertical')
    """

    sats, obsFile = parse.get_cei(nav, obs) # Satellite objects & opened observation file
    rec_loc = None
    if not pos == None: rec_loc = parse.get_rec_ascii(pos) # Ground Truth (if Position file provided)
    obsTime = parse.next_obs(sats, obsFile) # reciever time (datetime)
    xu = [0,0,0] # Initial User Position Estimate
    b = 0 # Initial Clock Error Estimate
    # Tropospheric correction values
    #alpha = [.1956E-07  , .2235E-07 , -.1192E-06 , -.1192E-06]
    #beta = [.1270E+06 ,  .1475E+06,  -.1966E+06 , -.1966E+06]
    result = {} # Data to Return
    iteration = 0
    while not obsTime == 0:
        SVs = [sats[sat] for sat in sats if sats[sat].tObs == obsTime] # Observed SVs
        rcvr_time = obsTime
        if (len(SVs) > 3) & (not rec_loc[iteration] == "NONE"):
            T0 = TGPS0 + timedelta(weeks = SVs[0].GPSWeek) # Init GPS week
            rcvr_tow = (rcvr_time - T0).total_seconds() # reciever time relative to GPSWeek
            pr = [] # pseudoranges
            for sat in SVs:
                """
                # Plotting Stuff
                if(iteration_valid >=150 and iteration_valid <=225):
                    y1points[sat.PRN].append(sat.pseudoRange)
                if(iteration_valid >=470 and iteration_valid <=530):
                    y2points[sat.PRN].append(sat.pseudoRange)
                """

                # Group Delay and Satellite Clock Bias correction
                dsv = satbias(sat, rcvr_tow)
                sat.pseudoRange = sat.pseudoRange + C*dsv  - C*sat.Tgd

                # Spoofing mechanism
                if (sat.PRN in pr_shifts): 
                    sat.pseudoRange = sat.pseudoRange + pr_shifts[sat.PRN][0] + pr_shifts[sat.PRN][1] * iteration
                '''
                #ionospheric
                if not sat.X == None: sat.pseudoRange -= C * iono(sat, alpha, beta, obsTime, xu)
        
                #tropospheric model
                if not sat.X == None:
                    LambdaU, PhiU, h = ecef2elli(xu[0],xu[1],xu[2])
                    lat = PhiU*180/np.pi
                    lng = LambdaU*180/np.pi
                    enu = pm.ecef2enu(sat.X, sat.Y, sat.Z, lat, lng, h)
                    E = np.arcsin(enu[2]/norm(enu))
                    sat.pseudoRange -= saast(xu, E)
                '''
                
            dx = [100, 100, 100] # difference in user position per iteration
            db = 100 # difference in user clock error per iteration
            while(norm(dx) > 0.1 and norm(db) > 1):
                Xs = [] # Satellite position matrix
                pr = [] # Corrected Pseudoranges
                for sat in SVs:
                    cpr = sat.pseudoRange - b # PR corrected for user clock bias
                    pr.append(cpr)
                    tau = cpr/C # transmission travel time
                    xs, ys, zs = sat_position(sat, rcvr_tow - tau) #satellite position
                    theta = omega_e * tau # Rotation correction
                    xs_vec = np.asarray([[math.cos(theta), math.sin(theta), 0],[-math.sin(theta), math.cos(theta), 0],[0,0,1]]) @ np.asarray([xs, ys, zs])
                    #xs_vec = [xs, ys, zs]
                    Xs.append(xs_vec)
                x_, b_, norm_dp, G = estimate_position(Xs, pr, len(SVs), xu, b)
                dx = x_ - xu
                db = b_ - b
                xu = x_
                b = b_
            lamb, phi, h = ecef2elli(xu[0],xu[1],xu[2])
            lat = phi*180/np.pi
            lon = lamb*180/np.pi
            err = None
            if not rec_loc == None: err = haversine(lat,lon, rec_loc[iteration][0], rec_loc[iteration][1]) * 1000
            result[obsTime] = {"X": xu[0], "Y": xu[1], "Z": xu[2], "SVs": SVs, "clockBias": b, "err": err}
            iteration_valid += 1
        iteration += 1
        obsTime = parse.next_obs(sats, obsFile) # reciever time (datetime)

    """
    ydpoints = {}
    for x in y1points:
        ydpoints[x] = np.subtract(y1points[x][0:len(y2points[x])], y2points[x])

    
    # Plotting Stuff
    axs[0,0].plot(ydpoints['G18'], label = 'Legitimate')
    axs[0,1].plot(ydpoints['G23'], label = 'Legitimate')
    axs[1,0].plot(ydpoints['G05'], label = 'Legitimate')
    axs[1,1].plot(ydpoints['G24'], label = 'Legitimate')
    axs[2,0].plot(ydpoints['G13'], label = 'Legitimate')
    axs[2,1].plot(ydpoints['G15'], label = 'Legitimate')
    axs[3,0].plot(ydpoints['G29'], label = 'Legitimate')
    axs[3,1].plot(ydpoints['G10'], label = 'Legitimate')
    """
    """
    axs[0,0].plot(y1points['G18'], label = 'Legitimate')
    axs[0,0].plot(y2points['G18'], label = 'Spoofed')
    axs[0,1].plot(y1points['G23'], label = 'Legitimate')
    axs[0,1].plot(y2points['G23'], label = 'Spoofed')
    axs[1,0].plot(y1points['G05'], label = 'Legitimate')
    axs[1,0].plot(y2points['G05'], label = 'Spoofed')
    axs[1,1].plot(y1points['G24'], label = 'Legitimate')
    axs[1,1].plot(y2points['G24'], label = 'Spoofed')
    axs[2,0].plot(y1points['G13'], label = 'Legitimate')
    axs[2,0].plot(y2points['G13'], label = 'Spoofed')
    axs[2,1].plot(y1points['G15'], label = 'Legitimate')
    axs[2,1].plot(y2points['G15'], label = 'Spoofed')
    axs[3,0].plot(y1points['G29'], label = 'Legitimate')
    axs[3,0].plot(y2points['G29'], label = 'Spoofed')
    axs[3,1].plot(y1points['G10'], label = 'Legitimate')
    axs[3,1].plot(y2points['G10'], label = 'Spoofed')

    fig.legend(*axs[0,0].get_legend_handles_labels(), loc='upper right')
    """

    return result
