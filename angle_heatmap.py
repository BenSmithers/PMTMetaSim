import sys 
import numpy  as np 
import matplotlib.pyplot as plt 

from otherdata import load_file

from math import pi 

if __name__=="__main__":

    test = load_file(sys.argv[1])


    prex = test[0]
    prey = test[1]
    prez = test[2]

    pre_pts = np.array([prex, prey, prez]).T
    unique_pre_pts = np.unique(pre_pts, axis=0)


    xpos = test[6]
    ypos = test[7]
    zpos = test[8]

    final_r = np.sqrt(xpos**2 + ypos**2)
    bad = np.isnan(final_r)

    vx = test[9]*-1
    vy = test[10]*-1
    vz = test[11]*-1


    # we need to get the direction along the dynode 

    # assume that the dynode is aligned along the Y-axis 

    
    dyn_x = -1*np.ones_like(xpos)
    dyn_x[xpos<0]*=-1 

    significance = np.ones_like(vx)
    should_eval = np.ones_like(dyn_x).astype(bool)


    these_thetas = np.arctan(dyn_x*vx[should_eval]/vz[should_eval])
    these_phis = np.abs(np.arctan(vx/vy))

    

    print("{} - {}".format(min(these_phis), max(these_phis)))

    angles = np.linspace(-pi/2, pi/2, 100)
    phipt = np.linspace(0, pi/2, len(angles)+1)

    data = np.histogram2d(these_thetas, these_phis, bins=(angles, phipt))[0]
    data = np.log(data + 1)
    plt.pcolormesh(angles*180/pi, phipt*180/pi, data.T, cmap='inferno')
    plt.xlabel("Zenith [deg]",size=14)
    plt.ylabel("Azimuth [deg]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("log( 1 + Counts)")
    plt.savefig("./plots/angular_heatmap.png", dpi=400)
    plt.show()