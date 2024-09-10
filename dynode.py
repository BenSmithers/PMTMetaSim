
import numpy as np 
import os 

import matplotlib.pyplot as plt 
from math import pi 
from utils import Irregular2DInterpolator
data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "data","dyn_new.dat"),
        delimiter=",",
        comments="#"
    ).T

thetas = data[0]*pi/180
phis = data[1]*pi/180
odds = (data[2] + data[3]*0.1 + data[4]*0.01)/(data[2]+data[3]+data[4])

print(odds)

i2d = Irregular2DInterpolator(
    thetas, phis, odds
)

angles = np.linspace(-pi/2, pi/2, 100)
phipt = np.linspace(0, pi/2, len(angles)+1)
data = i2d(angles, phipt, grid=True)

plt.pcolormesh(angles*180/pi, phipt*180/pi, data.T)
plt.xlabel("Zenith [deg]", size=14)
plt.ylabel("Azimith [deg]", size=14)
plt.scatter(thetas*180/pi, phis*180/pi, color='red', marker='o', label="Simulated")
plt.legend()
cbar  = plt.colorbar()
cbar.set_label("% On Dyn 1")
plt.show()

if False:
    plt.plot(thetas, data[1], label="Dynode 1")
    plt.plot(thetas, data[2], label="Dynode 2")
    plt.plot(thetas, data[3], label="Dynode 3")
    plt.ylabel("Hits")
    plt.xlabel("Angle [rad]")
    plt.legend()
    plt.show()
