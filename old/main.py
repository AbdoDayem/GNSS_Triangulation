import copy
from datetime import datetime, timedelta
import math

import numpy as np
from numpy.linalg import norm
from matlab import ecef2elli, estimate_position, iono, satbias, sat_position
from parse import get_cei, get_rec, next_obs
import matplotlib.pyplot as plt

from position import Satellite, cei2ecef, haversine, iterative, saast, tri_naive
import pymap3d as pm

# Constants
Y = (77/60)**2
TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time
c = 299792458
omega_e = 7.2921151467e-5


def pr_changes():
    # Adjustment Function
    def pr_adjust(adjustment):
        # File Processing
        sats, obsFile = get_cei("Field.23N", "Field.23O")
        rec_loc = get_rec("Field.pos")
        # Position Variables
        obsTime = 1
        xu = [0,0,0]
        b = 0
        xpoints = []
        ypoints = []
        zpoints = []
        while not obsTime == 0:
            rcvr_time = obsTime = next_obs(sats, obsFile)
            if obsTime in rec_loc:
                T0 = TGPS0 + timedelta(weeks = 2263) # Init GPS week
                rcvr_tow = (rcvr_time - T0).total_seconds()
                SVs = [sats[sat] for sat in sats if sats[sat].tObs == obsTime]
                pr = []
                for sat in SVs:
                    sat.pseudoRange += adjustment
                    xpoints.append(sat.pseudoRange)
                    dsv = satbias(sat, rcvr_tow)
                    sat.pseudoRange = sat.pseudoRange + c*dsv - c*sat.Tgd
                dx = [100, 100, 100]
                db = 100
                while(norm(dx) > 0.1 and db > 1):
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
                    if not (norm(dx) > 0.1 and db > 1):
                        for sat in SVs: 
                            cpr = sat.pseudoRange - b
                            tau = cpr/c
                            xs, ys, zs = sat_position(sat, rcvr_tow - tau, 1)
                            ypoints.append(tau)
                            zpoints.append([xs,ys,zs])
                lamb, phi, h = ecef2elli(xu[0],xu[1],xu[2])
                lat = phi*180/np.pi
                lon = lamb*180/np.pi
                
        return xpoints, ypoints, zpoints

    # Ground Truth
    gt_x, gt_y, gt_z = pr_adjust(0)

    ## Figure Setup
    # PR vs TSV
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_xlabel("Change in PR")
    ax1.set_ylabel("Change in TSV")

    # TSV vs SV X Coord
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_xlabel("Change in TSV")
    ax2.set_ylabel("Change in Sat X")

    # TSV vs SV Y Coord
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.set_xlabel("Change in TSV")
    ax3.set_ylabel("Change in Sat Y")

    # TSV vs SV Z Coord
    f4 = plt.figure()
    ax4 = f4.add_subplot(111)
    ax4.set_xlabel("Change in TSV")
    ax4.set_ylabel("Change in Sat Z")

    for i in range(100, 1001, 100):
        x,y,z = pr_adjust(i)
        xpoints = np.subtract(x, gt_x)
        ypoints = np.subtract(y, gt_y)
        zpoints = np.subtract(z, gt_z)
        ax1.scatter(xpoints, ypoints)
        ax2.scatter(ypoints, [i[0] for i in zpoints])
        ax3.scatter(ypoints, [i[1] for i in zpoints])
        ax4.scatter(ypoints, [i[2] for i in zpoints])

        print("Adjustment Complete:", i)

    # Plot
    plt.show()

def single_epoch(spoof):
    # Adjustment Function
    def pr_adjust(adjustment):
        # File Processing
        sats, obsFile = get_cei("Field.23N", "Field.23O")
        rec_loc = get_rec("Field.pos")
        # Position Variables
        obsTime = 1
        xu = [0,0,0]
        b = 0
        rcvr_time = obsTime = next_obs(sats, obsFile)
        if obsTime in rec_loc:
            T0 = TGPS0 + timedelta(weeks = 2263) # Init GPS week
            rcvr_tow = (rcvr_time - T0).total_seconds()
            SVs = [sats[sat] for sat in sats if sats[sat].tObs == obsTime]
            pr = []
            for sat in SVs:
                if sat.PRN == spoof: sat.pseudoRange += adjustment
                dsv = satbias(sat, rcvr_tow)
                sat.pseudoRange = sat.pseudoRange + c*dsv - c*sat.Tgd
            dx = [100, 100, 100]
            db = 100
            while(norm(dx) > 0.1 and db > 1):
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
        return lat,lon,h

    # Ground Truth
    gt_x, gt_y, gt_z = pr_adjust(0)

    ## Figure Setup

    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.set_xlabel("Change in PR (m)")
    ax3.set_ylabel("Change in lat/lon (m)")
    ax3.set_title("Sattelite:" + spoof)

    f4 = plt.figure()
    ax4 = f4.add_subplot(111)
    ax4.set_xlabel("Change in PR (m)")
    ax4.set_ylabel("Change in height (m)")
    ax4.set_title("Sattelite:" + spoof)

    i = 0.01
    while i < 10:
        x,y,z = pr_adjust(i)
        err = haversine(x, y, gt_x, gt_y) * 1000
        z_diff = np.subtract(z, gt_z)
        ax3.scatter(i, err)
        ax4.scatter(i, z_diff)
        i += 0.01

    # Plot
    plt.show()

def shift_x():
    obs_num = 20
    sats, obsFile = get_cei("Field.23N", "Field.23O")
    rec_loc = get_rec("Field.pos")
    c = 299792458
    omega_e = 7.2921151467e-5
    TGPS0 = datetime(year = 1980, month = 1, day = 6) # Relative Base GPS Time
    obsTime = 1
    p_points = []
    s_points = []
    gt_points = []
    j = 0
    for x in rec_loc:
        if j < obs_num: gt_points.append(pm.geodetic2ecef(rec_loc[x][0],rec_loc[x][1],rec_loc[x][2]))
        j+=1
    ypoints = []
    xu = [0,0,0]
    xu_c = [0,0,0]
    xu_s = [0,0,0]
    b = 0
    b_c = 0
    b_s = 0
    j = 0
    adjustment = 10
    user_clock_bias = []
    user_clock_bias_spoof = []
    while (j < obs_num) and (not obsTime == 0):
        j += 1
        rcvr_time = obsTime = next_obs(sats, obsFile)
        if obsTime in rec_loc:
            T0 = TGPS0 + timedelta(weeks = 2263) # Init GPS week
            rcvr_tow = (rcvr_time - T0).total_seconds()
            SVs = [sats[sat] for sat in sats if sats[sat].tObs == obsTime]
            SVcopy = copy.deepcopy(SVs)
            spoof = ["None", 0, 1000000000] # which sat to spoof
            for sat in SVs:
                obs_loc = pm.geodetic2ecef(rec_loc[obsTime][0], rec_loc[obsTime][1], rec_loc[obsTime][2])
                tau = sat.pseudoRange/c
                xs, ys, zs = sat_position(sat, rcvr_tow - tau, 1)

                d_x = xs - obs_loc[0]
                d_y = ys - obs_loc[1]
                d_h = zs - obs_loc[2]

                if (d_x - d_y) > (spoof[1] - spoof[2]): spoof = [sat.PRN, d_x, d_y]
            for i in range(2):
                pr = []
                if i == 1: 
                    SVs = SVcopy
                    b = b_s
                    xu = xu_s
                else:
                    b_s = copy.deepcopy(b_c)
                    #xu_s = copy.deepcopy(xu_c)
                    b = b_c
                    xu = xu_c
                for sat in SVs:
                    if i == 1: 
                        if sat.PRN == "G25": sat.pseudoRange += adjustment
                    dsv = satbias(sat, rcvr_tow)
                    sat.pseudoRange = sat.pseudoRange + c*dsv - c*sat.Tgd
                dx = [100, 100, 100]
                db = 100
                while(norm(dx) > 0.1 and db > 1):
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
                if i == 0:
                    p_points.append([xu[0],xu[1],xu[2]])
                    ypoints.append(err)
                    user_clock_bias.append(b)
                    xu_c = copy.deepcopy(xu)
                    b_c = copy.deepcopy(b)
                if i == 1:
                    user_clock_bias_spoof.append(b)
                    s_points.append([xu[0],xu[1],xu[2]])
                    xu_s = copy.deepcopy(xu)
                    b_s = copy.deepcopy(b)
        adjustment += 1
    
    origin = gt_points[0]
    print(origin)

    f5 = plt.figure()
    ax5 = f5.add_subplot(111)
    xpoints = [p[0] for p in gt_points]
    ypoints = [p[1] for p in gt_points]
    xpoints = np.subtract(xpoints, origin[0])
    ypoints = np.subtract(ypoints, origin[1])
    ax5.plot(xpoints,ypoints, label = "Ground Truth")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")

    xpoints = [p[0] for p in s_points]
    ypoints = [p[1] for p in s_points]
    xpoints = np.subtract(xpoints, origin[0])
    ypoints = np.subtract(ypoints, origin[1])
    print(len(ypoints))
    ax5.plot(xpoints,ypoints, '--', label = "Spoofed")

    xpoints = [p[0] for p in p_points]
    ypoints = [p[1] for p in p_points]
    xpoints = np.subtract(xpoints, origin[0])
    ypoints = np.subtract(ypoints, origin[1])
    print(len(ypoints))
    ax5.plot(xpoints,ypoints, ':', label = "Calculated")
    ax5.legend(loc='upper center')

    f6 = plt.figure()
    ax6 = f6.add_subplot(111)
    xpoints = []
    k = 0
    for time in rec_loc:
        if k < obs_num: xpoints.append(time)
        k += 1
    ypoints = user_clock_bias
    ax6.plot(xpoints,ypoints, label = "Calculated")
    ypoints = user_clock_bias_spoof
    ax6.plot(xpoints,ypoints, label = "Spoofed")
    ax6.legend(loc='upper center')



    plt.show()

shift_x()
