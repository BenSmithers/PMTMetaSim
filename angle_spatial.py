import sys 
import numpy  as np 
import matplotlib.pyplot as plt 

from otherdata import load_file

from math import pi 

g4bins  = np.arange(-0.35,0.35,0.025)

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)


if __name__=="__main__":

    test = load_file(sys.argv[1])


    prex = test[0]
    prey = test[1]
    prez = test[2]

    print(prex[:10])

    pre_pts = np.array([prex, prey, prez]).T
    unique_pre_pts = np.unique(pre_pts, axis=0)

    if len(test)>12:
        offset = 1
    else:
        offset = 0
    print("using offset {}".format(offset))

    xpos = test[6+offset]
    ypos = test[7+offset]
    zpos = test[8+offset]

    final_r = np.sqrt(xpos**2 + ypos**2)
    bad = np.isnan(final_r)

    vx = test[9+offset]*-1
    vy = test[10+offset]*-1
    vz = test[11 + offset]*-1

    # we need to get the direction along the dynode 

    # assume that the dynode is aligned along the Y-axis 

    
    dyn_x = -1*np.ones_like(xpos)
    dyn_x[xpos<0]*=-1 

    significance = np.ones_like(vx)
    should_eval = np.ones_like(dyn_x).astype(bool)


    these_thetas = np.arctan(dyn_x*vx[should_eval]/vz[should_eval])*180/pi
    these_phis = np.abs(np.arctan(vx/vy))*180/pi

    keep = np.logical_not(np.isnan(these_thetas))


    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=xpos[keep])[0]
    data /= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]

    plt.pcolormesh(bins, bins, data.T, vmin=-.015, vmax=0.015, cmap="RdBu")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("X Position on Dynode")
    plt.savefig("./plots/xpos_dynode_distrib.png", dpi=400)
    plt.show()

    plt.clf()
    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=these_thetas[keep])[0]
    data /= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]
    plt.pcolormesh(bins, bins, data.T, vmin=-90, vmax=90, cmap="coolwarm")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("Avg Zenith")
    plt.savefig("./plots/zenith_dynode_distrib.png", dpi=400)
    plt.show()


    plt.clf()
    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=these_phis[keep])[0]
    data/= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]

    plt.pcolormesh(bins, bins, data.T, vmin=0, vmax=90)
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("Avg Azimuth")
    plt.savefig("./plots/azimuth_dynode_distrib.png", dpi=400)
    plt.show()
