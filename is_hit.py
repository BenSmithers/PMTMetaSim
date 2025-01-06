from otherdata import load_file, get_color, set_axes_equal
import numpy as np 
import os 
from math import pi
import h5py as h5

from scipy.interpolate import interp1d, RectBivariateSpline
from utils import load_dyn_odds, new_dyn_odds, load_newest_dyn, perlin

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from load_dynode import build_interpolator

import pandas as pd

from pmtstupid import dumb_photonics


DEBUG = False

FUZZ = 1+perlin(61)*2/5

g4bins  = np.arange(-0.305,0.305,0.01)
print(len(g4bins))

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
binsy = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
pmt_costheta =  np.linspace(0, pi, 50)
pmt_azimuth = np.linspace(-pi, pi, 51)

pmt_a = 0.254564663
pmt_b = 0.254110205
pmt_c = 0.186002389

def spine_penalty_func2(xs):
    SCALE = 0.025
    base = np.sin(xs/SCALE)
    base[xs/SCALE > pi] = -1
    base[xs/SCALE < -pi] = -1 
    base += 1.0
    base /= 8.0
    base += 0.75
    return base 
    
def spine_penalty_func(xs):
    SCALE = 0.02
    dscale =np.exp(-(np.abs(xs)/SCALE)**1)
    return dscale 


def prepare_photonics(norm = False, wavelen=405):
    """
        Prepares and returns a PE-production tensor for particles generated at a given location 
    """
    _filename = "./processed_optics_{}.hdf5".format(wavelen)
    N_PHOT = 10000

    data = h5.File(_filename, 'r')

    initial_x = np.array(data["init_x"])/1000
    initial_y =  np.array(data["init_y"])/1000
    hit_x =  np.array(data["final_x"])/1000
    hit_y =  np.array(data["final_y"])/1000
    hit_z =  np.array(data["final_z"])/1000

    czen =np.arctan(hit_z/np.sqrt(hit_x**2 + hit_y**2))
    aziumuth = np.arctan2(hit_y, hit_x)

    czen_widths = pmt_costheta[1:] - pmt_costheta[:-1]
    azi_widths = pmt_azimuth[1:] - pmt_azimuth[:-1]
    czen_centers = 0.5*(pmt_costheta[1:] + pmt_costheta[:-1])
    
    print("{} - {}".format(min(czen), max(czen)))

    # histogram all of the data 
    photons = np.histogramdd(
        sample = (
            initial_x, initial_y, hit_x, hit_y
        ),
        bins = (
            bins, binsy, bins, binsy
        ),
    )[0]
    

    if DEBUG:
        phot_2d = np.histogram2d(hit_x, hit_y, bins=(bins, binsy))[0]
        plt.gca().set_aspect('equal')
        plt.pcolormesh(bins, binsy, phot_2d.T) #, vmin=0, vmax=10000)
        plt.xlabel("X [m]", size=14)
        plt.ylabel("Y [m]", size=14)
        plt.show()
    prenorm = np.zeros_like(photons)
    # now, we normalize by the number of photons simulated at each initail-value pair 
    for ix in range(len(photons)):
        for iy in range(len(photons[0])):
            
            
            prenorm[ix][iy] =photons[ix][iy]/N_PHOT
            
            if DEBUG : #and ix%11==0 and iy%11==0 :
                if np.sum(photons[ix][iy])<1:
                    continue
                
                # count good! 
                n_above =np.sum( (prenorm[ix][iy]>0).astype(int) )
                if n_above<120:
                    continue

                plt.pcolormesh(bins, binsy,np.log10(1+photons[ix][iy]))
                plt.title("{} - {}".format(ix, iy))
                plt.xlabel(r"$\theta_{hit}$ [rad]")
                plt.ylabel("$\phi_{hit}$ [rad]")
                plt.colorbar()
                plt.show()
            if norm:
                prenorm[ix][iy]/=np.sum(prenorm[ix][iy])

    return prenorm

_photon_tensor = prepare_photonics(False)


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

    interpo = build_interpolator()

    prex = test[0]
    prey = test[1]
    prez = test[2]

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


    spine_penalty = spine_penalty_func(xpos)
    spine_penalty /= np.max(spine_penalty)
    spine_penalty = 1-spine_penalty
    print("sp {} - {}".format(min(spine_penalty), max(spine_penalty)))





    these_thetas = np.arctan( np.sqrt(vx**2 + vy**2)/np.abs(vz))*180/pi
    these_thetas[ vy<0]*=-1


    
    # what if we actually manually shift the theta for spline-adjacet PEs 
    print("theta range {} - {}".format(these_thetas.min(), these_thetas.max()))
    
    # phi of 0 means all along x axis 
    these_phis = np.abs(np.arctan(np.abs(vx)/np.abs(vy)))*180/pi
    


    dyn_vector_z = np.ones_like(xpos)
    dyn_vector_y = 0*dyn_vector_z
    dyn_vector_x = np.ones_like(xpos)
    dyn_vector_z[xpos<0] = -1

    hit_angle = (dyn_vector_x*vx + dyn_vector_y*vy + dyn_vector_z*vz)/(np.sqrt(vx**2 + vy**2 + vz**2)*np.sqrt(dyn_vector_x**2 + dyn_vector_y**2 + dyn_vector_z**2))
    hit_angle = np.arccos(hit_angle)

    #hit_angle[hit_angle>90] = 180-hit_angle[hit_angle>90] 

    

    print( "phi range {} - {}".format(np.min(these_phis) , np.max(these_phis)))
    

    pzenith = (pi/2)-np.arctan(prez/np.sqrt(prex**2 + prey**2))
    pazimuth =np.arctan2(prey, prex)

    significance = interpo(xpos,ypos, these_thetas, these_phis)

    #sig_bin = np.histogram2d(pzenith, pazimuth, bins=(pmt_costheta, pmt_azimuth), weights=significance)[0]
    #counts =  np.histogram2d(pzenith, pazimuth, bins=(pmt_costheta, pmt_azimuth), weights=np.ones_like(significance))[0]
    sig_bin = np.histogram2d(prex, prey, bins=(bins, binsy), weights=significance)[0]
    counts =  np.histogram2d(prex, prey, bins=(bins, binsy))[0]
    sig_bin /= counts
    
    sig_bin /= np.nanmax(sig_bin)
     
    #det_odds = sig_bin*photon_tensor
    if True:
        det_odds = np.ones((len(bins)-1, len(binsy)-1))
        for ix in range(len(photon_tensor)):
            for iy in range(len(photon_tensor[ix])):
                det_odds[ix][iy] = np.nansum(photon_tensor[ix][iy]*sig_bin) #/np.nansum(counts*photon_tensor[ix][iy] )

    det_odds /= np.nanmax(det_odds)

    #sig_bin/=np.max(sig_bin)
    plt.clf()
    plt.pcolormesh(bins+0.417, binsy+0.297,det_odds.T, cmap='inferno', vmin=0, vmax=1)
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


    if False: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # we need to now average dots coming from the same place. 
        
        print("{} unique points, {} total".format(len(unique_pre_pts[0]), len(prex)))
        p = ax.scatter(unique_pre_pts[0], unique_pre_pts[1], unique_pre_pts[2], c=merged_significance, cmap="inferno") #, vmin=0.5, vmax=1.5)
        fig.colorbar(p, ax=ax)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        set_axes_equal(ax)
        plt.show()


if __name__=="__main__":
 
    #process("./data/somefield_highstat_2isheV.txt", "~mG")
    #process("./data/old_comsol_petraj/test_z500.txt", "z100")
    #process("./data/0mG_down_lamb.dat", "0mG")
    #process("./data/0mG_down_lamb_1eV.dat", "0mG")
    #process("./data/second_gen_data/0mG_lamb_1eV_fixv.dat", "lamb0")
    #process("./data/0mG_latest_lowest.dat", "low0")
    #process("./data/0mG_latest_higher.dat", "high_400", 400)
    #process("./data/0mG_latest_higher.dat", "high_540", 540)
    #process("./data/0mG_latest_lowest.dat", "high_365", 365)
    #process("./data/0mG_latest_lowest.dat", "high_410", 410)
    #process("./data/0mG_hemi_shuffle_0.5ev.dat", "high_540", 540)
    #process("./data/0mG_latest_lowest.dat", "high_410")
    
    baseline = process("./data/0mG_hemi_shuffle_0.5ev.dat", "0mG")
    #process("./data/250mg_z_latest.dat", "250mG_z") #)
    #process("./data/250mg_z_latest.dat", "250mG_z",normy=baseline)

    #process("./data/500ymG_hemi_shuffle_0.5ev.dat", "500mG_y")
    #process("./data/600mGxz_WIDE.dat", "600mG_xz")# )
    #process("./data/600mGy_WIDE.dat", "600mG_y")# )
    #process("./data/600mGx_WIDE.dat", "600mG_x")# )
    process("./data/250mg_z_latest.dat", label="250mG_z",normy=baseline)
    process("./data/250mg_y_latest.dat", label="250mG_y",normy=baseline)
    


