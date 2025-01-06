from utils import  SpecialInterpolator
from load_dynode import build_interpolator
import numpy as np 

from is_hit import prepare_photonics, load_file

from matplotlib import pyplot as plt 
from math import pi 

_photon_tensor = prepare_photonics(False)


g4bins  = np.arange(-0.305,0.305,0.01)

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
binsy = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)

def process(filename, label, reload_photonics = -1, normy=1):
    if False :
        photon_tensor = dumb_photonics(bins, binsy)
    else:
        if reload_photonics <0:
            photon_tensor = _photon_tensor  # get teh 
        else:
            photon_tensor = prepare_photonics(False, reload_photonics)

    test = load_file(filename) 

    yield_x = np.array([0,22.5,40, 50, 60, 90, 90+30, 90+40, 90+50, 90+90-22.5, 180 ])
    yield_x = 180.0 - yield_x
    yield_x*=pi/180
    yield_y = [1.3, 1.4, 1.6, 1.8, 1.88, 2, 1.88, 1.8, 1.6, 1.4, 1.3]

    dynpoints, dynvalues = build_interpolator()
    this_interpo = SpecialInterpolator(dynpoints.T, dynvalues)

    prex = test[0]
    prey = test[1]
    prez = test[2]

    if len(test)>12:
        offset = 1
    else:
        offset = 0
    print("using offset {}".format(offset))

    xpos = test[6+offset]
    ypos = test[7+offset]
    zpos = test[8+offset]

    vx = test[9+offset]*-1
    vy = test[10+offset]*-1
    vz = test[11 + offset]*-1

    phis = np.abs(np.arctan(np.abs(vx)/np.abs(vy)))*180/pi
    thetas = np.arctan( np.sqrt(vx**2 + vy**2)/np.abs(vz))*180/pi
    thetas[vx>0]*=-1



    significance =[this_interpo(xpos[i], ypos[i], thetas[i], phis[i]) for i in range(len(thetas))]
    sig_bin = np.histogram2d(prex, prey, bins=(bins, binsy), weights=significance)[0]
    counts =  np.histogram2d(prex, prey, bins=(bins, binsy))[0]
    sig_bin /= counts
    
    sig_bin /= np.nanmax(sig_bin)
    if True:
        det_odds = np.ones((len(bins)-1, len(binsy)-1))
        for ix in range(len(photon_tensor)):
            for iy in range(len(photon_tensor[ix])):
                det_odds[ix][iy] = np.nansum(photon_tensor[ix][iy]*sig_bin) #/np.nansum(counts*photon_tensor[ix][iy] )

    det_odds /= np.nanmax(det_odds)

    #sig_bin/=np.max(sig_bin)
    plt.clf()
    plt.pcolormesh(bins+0.417, binsy+0.297,sig_bin.T, cmap='inferno', vmin=0, vmax=1)
    #plt.pcolormesh(bins, binsy, 0-sig_bin.T, cmap='RdBu') #, vmin=0, vmax=1)
    plt.xlabel("X [m]", size=14)
    
    
    plt.title("Relative Avg. Charge",size=14)
    plt.ylabel("Y [m]")
    cbar = plt.colorbar()
    cbar.set_label("Relative Avg. Charge.")
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.savefig("./plots/det_eff_{}.png".format(label), dpi=400)
    plt.show()

    return det_odds

if __name__=="__main__":
    baseline = process("./data/0mG_hemi_shuffle_0.5ev.dat", "0mG")
    #process("./data/250mg_z_latest.dat", label="250mG_z",normy=baseline)
    #process("./data/250mg_y_latest.dat", label="250mG_y",normy=baseline)
    process("./data/0mG_latest_lowest.dat", "low0")