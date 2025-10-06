from datetime import datetime, timedelta
import math

import numpy as np
from numpy.linalg import norm
from matlab import ecef2elli, estimate_position, iono, satbias, sat_position
from parse import get_cei, get_rec, next_obs
import matplotlib.pyplot as plt

from position import Satellite, cei2ecef, haversine, iterative, saast, tri_naive
import pymap3d as pm


def test_cei2ecef():
    ## Test
    af0 = -5.03109768033e-05 # SVClockBias
    af1 = -1.477928890381e-12 # SVClockDrift
    af2 = 0 # SVClockDriftRate
    Tsv = 518384 # transmission time
    Toe = 518384
    toc = datetime(2021,12,31,23,59,44)
    GPSWeek = 2190
    e = 0.007046940154396 # Eccentricity
    sqrtA= 5153.70575141
    Cic= 1.024454832077e-7
    Crc= 369.90625
    Cis= 1.620501279831e-7
    Crs= 81
    Cuc= 4.43309545517e-6
    Cus= 5.327165126801e-7
    DeltaN= 4.677337687032e-9
    Omega0= -2.116071249787
    omega= 0.07183725937306
    Io= 0.9651958584989
    OmegaDot= -8.526783746197e-9
    IDOT= 5.571660653459e-11
    M0= 1.685062232029

    sat = Satellite(af0, af1, af2, Tsv, Toe, toc, GPSWeek, e, sqrtA, Cic, Crc, Cis, Crs, Cuc, Cus, DeltaN, Omega0, omega, Io, OmegaDot, IDOT, M0)
    cei2ecef(sat)
        # Geodetic Coordinates - Using pymap3d
    lat, lon, alt = pm.ecef2geodetic(sat.X, sat.Y, sat.Z)
    print("lat:", lat)
    print("lon:", lon)
    print("alt:", alt)
    
    print("X:", sat.X)
    print("Y:", sat.Y)
    print("Z:", sat.Z)

def test_triangulate():
    sat1 = Satellite(X=1, Y=1, Z=1, pseudoRange=0)
    sat2 = Satellite(X=1, Y=1, Z=0, pseudoRange=1)
    sat3 = Satellite(X=0, Y=1, Z=1, pseudoRange=1)
    sat4 = Satellite(X=1, Y=0, Z=1, pseudoRange=1)
    sat5 = Satellite(X=0, Y=0, Z=1, pseudoRange=np.sqrt(2))
    tri_naive(sat1, sat2, sat3, sat4, sat5)

def test_she2():
    ## Test
    # sat variables to calculate satellite positions
    cei13 = Satellite(af0 = 0.000238197, af1 = 5.91E-12, af2 = 0, Tsv = 513762,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.005791032,
        sqrtA= 5153.663012, Cic= -5.40E-08, Crc= 275.84375, Cis= 2.79E-08,
        Crs= 6.84375, Cuc= 2.91E-07, Cus= 5.55E-06, DeltaN= 4.82E-09,
        Omega0= 1.179072475, omega= 0.976888338, Io= 0.968071729, OmegaDot= -8.08E-09,
        IDOT= 1.88E-10, M0= 1.053777735, Tgd = -1.12E-08)
    cei23 = Satellite(af0 = 1.59E-05, af1 = -3.98E-12, af2 = 0, Tsv = 513048,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.001952713,
        sqrtA= 5153.698454, Cic= -4.47E-08, Crc= 158.21875, Cis= -2.24E-08,
        Crs= -83.65625, Cuc= -4.29E-06, Cus= 1.18E-05, DeltaN= 3.94E-09,
        Omega0= -0.028740278, omega= 2.895011309, Io= 0.967294461, OmegaDot= -7.52E-09,
        IDOT= 4.80E-10, M0= -1.259886663, Tgd = -8.38E-09)
    cei30 = Satellite(af0 = -0.00050351, af1 = -2.73E-12, af2 = 0, Tsv = 511218,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.005382175,
        sqrtA= 5153.592802, Cic= 1.17E-07, Crc= 197.15625, Cis= -3.17E-08,
        Crs= 0.03125, Cuc= -1.40E-07, Cus= 8.62E-06, DeltaN= 5.18E-09,
        Omega0= 2.113154349, omega= -2.75109377, Io= 0.935904972, OmegaDot= -8.14E-09,
        IDOT= -5.47E-10, M0= -1.281965678, Tgd = 3.73E-09)
    cei17 = Satellite(af0 = 0.000555238, af1 = 5.12E-12, af2 = 0, Tsv = 516258,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.013756476,
        sqrtA= 5153.729422, Cic= 9.69E-08, Crc= 393.90625, Cis= -4.04E-07,
        Crs= 47.9375, Cuc= 2.11E-06, Cus= -3.33E-07, DeltaN= 4.36E-09,
        Omega0= -2.034827297, omega= -1.51428616, Io= 0.980198565, OmegaDot= -8.30E-09,
        IDOT= 8.21E-11, M0= 1.59236821, Tgd = -1.12E-08)
    cei21 = Satellite(af0 = 0.0001551, af1 = 9.09E-13, af2 = 0, Tsv = 511218,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.024433852,
        sqrtA= 5153.609568, Cic= 4.21E-07, Crc= 330.34375, Cis= 2.05E-08,
        Crs= -132, Cuc= -7.11E-06, Cus= 2.06E-06, DeltaN= 4.42E-09,
        Omega0= -1.126305214, omega= -1.035171512, Io= 0.958981118, OmegaDot= -8.05E-09,
        IDOT= -5.06E-10, M0= 1.739378448, Tgd = -1.02E-08)
    cei27 = Satellite(af0 = 4.03E-05, af1 = 2.54E-11, af2 = 0, Tsv = 511218,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.009935614,
        sqrtA= 5153.651279, Cic= -1.62E-07, Crc= 377.96875, Cis= 1.96E-07,
        Crs= 81.8125, Cuc= 4.27E-06, Cus= 4.82E-07, DeltaN= 4.54E-09,
        Omega0= -2.097011575, omega= 0.628104458, Io= 0.973858491, OmegaDot= -8.47E-09,
        IDOT= 1.09E-10, M0= 1.602968087, Tgd = 1.86E-09)
    cei08 = Satellite(af0 =-5.03E-05, af1 = -1.48E-12, af2 = 0, Tsv = 513048,
        Toe = 518384, toc = datetime(2021,12,31,23,59,44) , GPSWeek = 2190, e = 0.00704694,
        sqrtA= 5153.705757, Cic= 1.02E-07, Crc= 369.90625, Cis= 1.62E-07,
        Crs= 81, Cuc= 4.43E-06, Cus= 5.33E-07, DeltaN= 4.68E-09,
        Omega0= -2.11607125, omega= 0.071837259, Io= 0.965195858, OmegaDot= -8.53E-09,
        IDOT= 5.57E-11, M0= 1.685062232, Tgd = 5.12E-09)
    cei01 = Satellite(af0 = 0.000469127, af1 = -1.00E-11, af2 = 0, Tsv = 512736,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.011218139,
        sqrtA= 5153.674995, Cic= -3.17E-08, Crc= 299.75, Cis= 1.96E-07,
        Crs= -141.125, Cuc= -7.36E-06, Cus= 4.70E-06, DeltaN= 3.99E-09,
        Omega0= -1.03661124, omega= 0.884087602, Io= 0.986418769, OmegaDot= -8.13E-09,
        IDOT= -3.78E-10, M0= -0.624294238, Tgd = 5.12E-09)
    cei07 = Satellite(af0 = 0.00029715, af1 = 3.98E-12, af2 = 0, Tsv = 511218,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.015393639,
        sqrtA= 5153.702707, Cic= 3.91E-07, Crc= 206.5, Cis= 1.60E-07,
        Crs= -0.875, Cuc= 5.40E-08, Cus=8.45E-06, DeltaN= 4.84E-09,
        Omega0= 2.099309531, omega=-2.297081704, Io= 0.950722051, OmegaDot= -7.95E-09,
        IDOT= -5.55E-10, M0= -1.281304151, Tgd = -1.12E-08)
    cei14 = Satellite(af0 = -6.40E-05, af1 = -5.68E-12, af2 = 0, Tsv = 513042,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.00126703,
        sqrtA= 5153.651503, Cic= -1.86E-09, Crc= 190.53125, Cis= 3.73E-09,
        Crs= 144.96875, Cuc=7.50E-06, Cus=9.76E-06, DeltaN=4.39E-09,
        Omega0= -3.1024374, omega= 3.064523682, Io= 0.954764368, OmegaDot= -7.88E-09,
        IDOT= -2.43E-11, M0= -1.935891005, Tgd = -7.92E-09)
    cei10 = Satellite(af0 = -0.000282293, af1 = -9.32E-12, af2 = 0, Tsv = 514266,
        Toe = 518400, toc = datetime(2022,1,1,0,0,0) , GPSWeek = 2190, e = 0.007406066,
        sqrtA= 5153.682199, Cic= 4.66E-08, Crc=155.65625, Cis=4.66E-08,
        Crs= -83.5, Cuc=-4.37E-06, Cus=1.20E-05, DeltaN= 3.81E-09,
        Omega0= -0.004130832, omega= -2.546675481, Io=0.972251248, OmegaDot=-7.39E-09,
        IDOT= 5.39E-10, M0= -2.61960098, Tgd = 2.33E-09)

    # pseudoranges from observation files
    sat13 = Satellite(pseudoRange=24353997.371)
    sat23 = Satellite(pseudoRange=25772495.523)
    sat30 = Satellite(pseudoRange=21256793.375)
    sat17 = Satellite(pseudoRange=24118134.188)
    sat21 = Satellite(pseudoRange=20808391.609)
    sat27 = Satellite(pseudoRange=23866854.086)
    sat08 = Satellite(pseudoRange=21378271.539)
    sat01 = Satellite(pseudoRange=21263253.660)
    sat07 = Satellite(pseudoRange=20975970.211)
    sat14 = Satellite(pseudoRange=22586435.523)
    sat10 = Satellite(pseudoRange=24990674.609)
    # get satellite positions & SV clock errors

    pairs = [[sat17, sat21, sat23, sat30, sat13, sat27, sat08, sat01, sat07, sat14, sat10],
             [cei17, cei21, cei23, cei30, cei13, cei27, cei08, cei01, cei07, cei14, cei10]]
    # triangulate using least squares & iterative method
    TRec = datetime(2022,1,1)
    x,y,z,t = iterative(pairs, TRec)

    # Convert to lat/lon
    lat, lon, alt = pm.ecef2geodetic(x, y, z)
    print("X:",x,"\nY:",y,"\nZ:",z)
    print("lat:",lat,"\nlon:",lon,"\nalt:",alt)
    print("Receiver Err:", t, "s")
    print("ERR:", haversine(lat, lon, 46.220698368, -64.552000945) * 1000, "m")

def test_field():
    ## Test

    sats, obsFile = get_cei("Field.23N", "Field.23O")
    sat25 = sats["G25"]
    sat09 = sats["G09"]
    sat29 = sats["G29"]
    sat05 = sats["G05"]
    sat20 = sats["G20"]
    sat12 = sats["G12"]
    sat11 = sats["G11"]
    sat06 = sats["G06"]

    sat25.pseudoRange=21978389.702
    sat09.pseudoRange=24571459.470
    sat29.pseudoRange=24178410.179
    sat05.pseudoRange=20939387.833
    sat20.pseudoRange=20166391.899
    sat12.pseudoRange=21196861.060
    sat11.pseudoRange=20929818.414
    sat06.pseudoRange=23067844.440


    sats = [sat25, sat09, sat29, sat05, sat20, sat12, sat11, sat06]
    # triangulate using least squares & iterative method
    TRec = datetime(2023,5,25,19,43,15)
    x,y,z,t = iterative(sats, TRec)

    # Convert to lat/lon
    lat, lon, alt = pm.ecef2geodetic(x, y, z)
    print("X:",x,"\nY:",y,"\nZ:",z)
    print("lat:",lat,"\nlon:",lon,"\nalt:",alt)
    print("Receiver Err:", t, "s")
    print("ERR:", haversine(lat, lon, 33.215943891, -87.531586815) * 1000, "m")

    lat, lon, alt = pm.ecef2geodetic(x, y, z)
    print(lat,lon,alt)

def test_parse():
    sats, obsFile = get_cei("Field.23N", "Field.23O")
    obsTime = next_obs(sats, obsFile)

def test_all():
    ## Test
    sats, obsFile = get_cei("Field.23N", "Field.23O")
    rec_loc = get_rec("Field.pos")
    obsTime = 1
    ypoints = []
    min = 100000
    max = 0
    while not obsTime == 0:
        obsTime = next_obs(sats, obsFile)
        if obsTime in rec_loc:
            array = [sats[sat] for sat in sats if sats[sat].tObs == obsTime]
            x,y,z,t = iterative(array, obsTime)
            lat, lon, alt = pm.ecef2geodetic(x, y, z)
            err = haversine(lat, lon, rec_loc[obsTime][0], rec_loc[obsTime][1]) * 1000
            if err > max: max = err
            if err < min: min = err
            ypoints.append(err)
            print("Observation:", obsTime, "ERR:", err, "m")
    xpoints = [time for time in rec_loc]
    print("MIN ERR:", min)
    print("MAX ERR:", max)

    plt.scatter(xpoints,ypoints)
    plt.show()

    plt.plot(xpoints,ypoints)
    plt.show()
        
def test_mat():
    sats, obsFile = get_cei("Field.23N", "Field.23O")
    rec_loc = get_rec("Field.pos")
    c = 299792458
    omega_e = 7.2921151467e-5
    Y = (77/60)**2
    TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time
    obsTime = 1
    xu = [0,0,0]
    alpha = [.1956E-07,  .2235E-07,  -.1192E-06,  -.1192E-06]
    beta = [.1270E+06,   .1311E+06,  -.1966E+06,  -.1966E+06]
    b = 0
    ypoints = []
    xpoints = []
    min = 100000
    max = 0
    user_clock_bias = []
    while not obsTime == 0:
        rcvr_time = obsTime = next_obs(sats, obsFile)
        if obsTime in rec_loc:
            T0 = TGPS0 + timedelta(weeks = 2263) # Init GPS week
            rcvr_tow = (rcvr_time - T0).total_seconds()
            SVs = [sats[sat] for sat in sats if sats[sat].tObs == obsTime]
            pr = []
            for sat in SVs:
                dsv = satbias(sat, rcvr_tow)
                #ionospheric and group delay IFF l2c pseudorange available
                #if not sat.l2c == None:
                #sat.pseudoRange = (sat.l2c - sat.pseudoRange * Y)/(1-Y)
                #ionospheric and group delay if no l2c pseudorange available
                #else: 
                #if not sat.X == None: sat.pseudoRange -= c * iono(sat, alpha, beta, obsTime, xu)
                sat.pseudoRange = sat.pseudoRange + c*dsv - c*sat.Tgd
                #tropospheric model
                if False: #not sat.X == None:
                    LambdaU, PhiU, h = ecef2elli(xu[0],xu[1],xu[2])
                    lat = PhiU*180/np.pi
                    lng = LambdaU*180/np.pi
                    enu = pm.ecef2enu(sat.X, sat.Y, sat.Z, lat, lng, h)
                    E = np.arcsin(enu[2]/norm(enu))
                    sat.pseudoRange -= saast(xu, E)
            dx = [100, 100, 100]
            db = 100
            while(norm(dx) > 0.1 and norm(db) > 1):
                Xs = []
                pr = []
                for sat in SVs:
                    cpr = sat.pseudoRange - b
                    pr.append(cpr)
                    tau = cpr/c
                    xs, ys, zs = sat_position(sat, rcvr_tow - tau, 1)
                    theta = omega_e * tau
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
            err = haversine(lat, lon, rec_loc[obsTime][0], rec_loc[obsTime][1]) * 1000
            if err > max: max = err
            if err < min: min = err
            ypoints.append(err)
            xpoints.append(len(SVs))
            user_clock_bias.append(b)
            print("Observation:", obsTime, "ERR:", err, "m")
   
    print("MIN:", min, "m")
    print("MAX:", max, "m")
    print("MEAN:", sum(ypoints)/len(ypoints), "m")
    print("MEDIAN:", np.median(ypoints), "m")

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(xpoints,ypoints, s = 1)
    ax1.set_xlabel("Number of SVs")
    ax1.set_ylabel("Error (meters)")
    ax1.set_title("Error vs Number of SVs")

    xpoints = [time for time in rec_loc]
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.scatter(xpoints,ypoints, s = 1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error (meters)")
    ax2.set_title("Error vs Time")

    f6 = plt.figure()
    ax6 = f6.add_subplot(111)
    xpoints = xpoints = [time for time in rec_loc]
    ypoints = user_clock_bias
    ax6.plot(xpoints,ypoints, label = "Ground Truth")


    plt.show()
    
test_mat()