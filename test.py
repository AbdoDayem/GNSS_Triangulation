
from matplotlib import pyplot as plt
import numpy as np
import pymap3d as pm
from parse import get_rec, get_rec_ascii
from position import position

start = 0
end = 20

rec_loc = get_rec_ascii("data/New.ASCII")
results = position("data/New.23N", "data/New.23O", "data/New.ASCII")
#results = position("data/Static.23N", "data/Static.23O", "data/Static.pos")
#spoofed = position("data/Field.23N", "data/Field.23O", "data/Field.pos", {"G25":[10, 0], "G10":[100,0]})
#rec_loc = get_rec("data/Static.pos")

gt_points = [[x[0], x[1]] for x in rec_loc if not x == "NONE"]
xpoints = [obsTime for obsTime in results]
ypoints = [results[obsTime]["err"] for obsTime in results]

"""
print(len(ypoints))
plt.scatter(xpoints, ypoints, s=1)
plt.xlabel("Time")
plt.ylabel("Error (m)")
"""

print("MIN:", np.min(ypoints), "m")
print("MAX:", np.max(ypoints), "m")
print("MEAN:", sum(ypoints)/len(ypoints), "m")
print("MEDIAN:", np.median(ypoints), "m")
start = 0

"""
f6 = plt.figure()
ax6 = f6.add_subplot(111)
xpoints = [results[obsTime]["numSV"] for obsTime in results]
ax6.scatter(xpoints,ypoints)
ax6.set_xlabel("# of Satellites")
ax6.set_ylabel("Error (m)")
"""

"""
f5 = plt.figure()
ax5 = f5.add_subplot(111)
geodetic = [pm.ecef2geodetic(results[obsTime]["X"],results[obsTime]["Y"],results[obsTime]["Z"]) for obsTime in results]
xpoints = [x[0] for x in geodetic][150:225]
ypoints = [x[1] for x in geodetic][150:225]
ax5.plot(xpoints,ypoints, label = "Legitimate")
xpoints = [x[0] for x in geodetic][470:530]
ypoints = [x[1] for x in geodetic][470:530]
ax5.plot(xpoints,ypoints, label = "Spoofed")
ax5.set_xlabel("lat")
ax5.set_ylabel("lon")
"""
f5 = plt.figure()
ax5 = f5.add_subplot(111)
geodetic = [pm.ecef2geodetic(results[obsTime]["X"],results[obsTime]["Y"],results[obsTime]["Z"]) for obsTime in results]
ypoints = [results[obstime]['clockBias'] for obstime in results][150:225]
print(ypoints)
ax5.plot(ypoints, label = "Legitimate")
ypoints = [results[obstime]['clockBias'] for obstime in results][470:530]
ax5.plot(ypoints, label = "Spoofed")
ax5.set_title("User Clock Bias over Time")
ax5.set_xlabel("Time")
ax5.set_ylabel("User Clock Bias (s)")

"""
for sat in sats:
    f7 = plt.figure()
    ax7 = f7.add_subplot(111)
    xpoints = range(75)
    ypoints = [results[obsTime]["SVs"] for obsTime in results if sat in results[obsTime]["SVs"]][150:225]
    ax7.plot(xpoints,ypoints, label = "Legitimate")
    xpoints = range(60)
    ypoints = [sat.pseudorange for obsTime in results if sat in results[obsTime]["SVs"]][470:530]
    ax7.plot(xpoints,ypoints, label = "Spoofed")

    ax7.set_title(sat.PRN)
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Pseudorange (m)")
    ax7.legend(loc='upper left')
"""

"""
xpoints = [p[0] for p in gt_points]
ypoints = [p[1] for p in gt_points]
ax5.plot(xpoints,ypoints, label = "Ground Truth")
ax5.set_xlabel("latitude")
ax5.set_ylabel("longitude")
"""
ax5.legend(loc='lower right')

"""
geodetic = [pm.ecef2geodetic(spoofed[obsTime]["X"],spoofed[obsTime]["Y"],spoofed[obsTime]["Z"]) for obsTime in spoofed][start:end]
xpoints = [x[0] for x in geodetic]
ypoints = [x[1] for x in geodetic]
ax5.plot(xpoints,ypoints, '--', label = "Spoofed")

ax5.legend(loc='upper center')

"""

plt.show()
