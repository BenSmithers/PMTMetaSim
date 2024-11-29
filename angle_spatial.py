import sys 
import numpy  as np 
import matplotlib.pyplot as plt 

from otherdata import load_file

from math import pi 

g4bins  = np.arange(-0.35,0.35,0.0125)

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)

def spine_penalty_func(xs):
    SCALE = 0.02
    dscale =np.exp(-(np.abs(xs)/SCALE)**1)
    return dscale 

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


    dyn_vector_z = np.ones_like(xpos)
    dyn_vector_y = 0*dyn_vector_z
    dyn_vector_x = np.ones_like(xpos)
    dyn_vector_z[xpos<0] = -1
    hit_angle = (dyn_vector_x*vx + dyn_vector_y*vy + dyn_vector_z*vz)/(np.sqrt(vx**2 + vy**2 + vz**2)*np.sqrt(dyn_vector_x**2 + dyn_vector_y**2 + dyn_vector_z**2))
    hit_angle = np.arccos(hit_angle)*180/pi 
    hit_angle[hit_angle>90] = 180-hit_angle[hit_angle>90] 

    print("hit angle range {} - {}".format(min(hit_angle), max(hit_angle)))

    significance = np.ones_like(vx)

    

    these_thetas = np.arctan(np.abs(vx)/ np.abs(vz)) 

    # positive side and moving towards negative
    # or negative side and moving towards positive 
    negative = np.logical_or(np.logical_and(xpos<0, vx>0), np.logical_and( xpos>0, vx<0 ))
    these_thetas[negative]*=-1

    spine_penalty = spine_penalty_func(xpos)
    spine_penalty /= np.max(spine_penalty)
    spine_penalty = 1-spine_penalty
    
    these_thetas = these_thetas*(1-spine_penalty) + 0.5*(spine_penalty*pi/2 + these_thetas)

    #these_thetas = these_thetas*(1-spine_penalty) + 0.5*(spine_penalty*pi/2 + these_thetas)

    these_phis = np.abs(np.arctan(vx/vy))*180/pi

    keep = np.logical_not(np.isnan(these_thetas))

    these_thetas*=90


    plt.clf()
    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=spine_penalty[keep])[0]
    data /= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]
    plt.pcolormesh(bins, bins, data.T )#, vmin=-90, vmax=90, cmap="coolwarm")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("Avg Zenith")
    plt.savefig("./plots/zenith_dynode_distrib.png", dpi=400)
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

    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=xpos[keep])[0]
    data /= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]
    plt.clf()
    plt.pcolormesh(bins, bins, data.T, vmin=-.015, vmax=0.015, cmap="RdBu")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("X Position on Dynode")
    plt.savefig("./plots/xpos_dynode_distrib.png", dpi=400)
    plt.show()

    plt.clf()
    data = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=hit_angle[keep])[0]
    data /= np.histogram2d(prex[keep], prey[keep], bins=(bins, bins))[0]
    plt.pcolormesh(bins, bins, data.T, vmin=-90, vmax=90, cmap="coolwarm")
    plt.xlabel("X [m]",size=14)
    plt.ylabel("Y [m]", size=14)
    lbl = plt.colorbar()
    lbl.set_label("Avg Hit Angle")
    plt.savefig("./plots/hit_distrib.png", dpi=400)
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
