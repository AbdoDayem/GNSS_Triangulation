
from __future__ import annotations
from datetime import datetime, timedelta
import itertools
import math
import pymap3d as pm
import numpy as np
from numpy.linalg import norm, inv
from numpy import dot
from dataclasses import dataclass
import scipy
from scipy.optimize import fsolve
from scipy.optimize import leastsq

# Constants that we will need
# Speed of light
C = 299792458
# Earth's rotation rate
omega_e = 7.2921151467e-5
TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time


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

def estimate_position(xs, pr, numSat, x0, b0,dim = 3):
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
        dp = np.subtract(pr, norms) + b - b0
        dp = np.add(dp, b)
        dp = np.subtract(dp, b0)
        array = [1] * numSat
        G = np.multiply(diff, -1)
        for i in range(len(norms)):
            G[i] = np.divide(G[i], norms[i])
        G = [np.append(row,1) for row in G]
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ dp
        
        dx = np.transpose([sol[0],sol[1],sol[2]])
        db = sol[dim]

        norm_dp = norm(dp)
        numIter = numIter + 1
        x0 = x0 + dx
        b0 = b0 + db
    return x0, b0, norm_dp, G

def sat_position(eph, t, harmonic = 0):
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
    #Ek = math.acos((e  + math.cos(nu))/(1+e*math.cos(nu)))
 
    Phi = nu + eph.omega
    du = 0
    dr = 0
    di = 0
    if (harmonic == 1):  
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
    Ek = fsolve(f, mk)
    dsv = sat.af0 + sat.af1 * (t-Toc) + sat.af2 * (t-Toc)**2 + F * sat.e * sat.sqrtA * math.sin(Ek)
    return dsv

def iono(sat, alpha, beta, GPSTime, xu):
    LambdaU, PhiU, h = ecef2elli(xu[0],xu[1],xu[2])
    lat = PhiU*180/np.pi
    lng = LambdaU*180/np.pi
    enu = pm.ecef2enu(sat.X, sat.Y, sat.Z, lat, lng, h)
    A = np.arctan2(enu[0], enu[1])
    E = np.arcsin(enu[2]/norm(enu))
    psi = (0.0137/(E + 0.11)) - 0.022
    F = 1.0 + 16.0*(0.53-E)**3
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

