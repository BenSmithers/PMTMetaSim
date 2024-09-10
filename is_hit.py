from otherdata import load_file, get_color, set_axes_equal
import numpy as np 
import os 
from math import pi
import h5py as h5
from scipy.interpolate import interp1d, RectBivariateSpline
from utils import Irregular2DInterpolator

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import pandas as pd


DEBUG = False


g4bins  = np.arange(-0.305,0.305,0.01)

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
binsy = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
pmt_costheta =  np.linspace(0, 1, 30)
pmt_azimuth = np.linspace(-pi, pi, 31)

pmt_a = 0.254564663
pmt_b = 0.254110205
pmt_c = 0.186002389

def prepare_photonics():
    """
        Prepares and returns a PE-production tensor for particles generated at a given location 
    """
    _filename = "./processed_optics.hdf5"
    N_PHOT = 10000

    data = h5.File(_filename, 'r')

    initial_x = np.array(data["init_x"])/1000
    initial_y =  np.array(data["init_y"])/1000
    hit_x =  np.array(data["final_x"])/1000
    hit_y =  np.array(data["final_y"])/1000
    hit_z =  np.array(data["final_z"])/1000

    zenith = np.cos(np.arctan(hit_z/np.sqrt(hit_x**2 + hit_y**2)))
    aziumuth = np.arctan2(hit_y, hit_x)

    # histogram all of the data 
    photons = np.histogramdd(
        sample = (
            initial_x, initial_y, hit_x, hit_y
        ),
        bins = (
            bins, binsy, bins, bins
        )
    )[0]
    prenorm = np.zeros_like(photons)
    # now, we normalize by the number of photons simulated at each initail-value pair 
    for ix in range(len(photons)):
        for iy in range(len(photons[0])):
            

            prenorm[ix][iy] = photons[ix][iy]/N_PHOT
            if DEBUG and ix%10==0 and iy%10==0:
                if np.sum(photons[ix][iy])<1:
                    continue
                plt.pcolormesh(bins, bins,np.log10(photons[ix][iy]+1).T)
                plt.xlabel("Cos(theta)")
                plt.ylabel("Asimuth [rad]")
                plt.colorbar()
                plt.show()

    return prenorm

photon_tensor = prepare_photonics()

def load_dyn_odds():
    data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "data","dyn_sim_geo.dat"),
        delimiter=",",
        comments="#"
    ).T 

    thetas = -1*np.arctan(data[1]/data[0])
    odds = (data[2] + data[3]*0.1+ data[4]*0.01)/(data[2]+data[3]+data[4]) # odds of hitting first dynode

    return thetas, odds 

def new_dyn_odds():
    data = np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data","dyn_new.dat"),
            delimiter=",",
            comments="#"
        ).T

    thetas = data[0]*pi/180
    phis = data[1]*pi/180
    odds = (data[2] + data[3]*0.1 + data[4]*0.01)/(data[2]+data[3]+data[4])


    i2d = Irregular2DInterpolator(
        thetas, phis, odds
    )
    return i2d

def process(filename, label):
    test = load_file(filename) 

    interpo = new_dyn_odds()

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

    keep = np.ones_like(these_thetas).astype(bool)

    these_odds = interpo(these_thetas, these_phis, grid=False)
    these_odds /= interpo(0,0)

    significance[should_eval] = these_odds

    significance[bad] = 0.0

    pzenith = np.cos(np.arctan(prez/np.sqrt(prex**2 + prey**2))[keep])
    pazimuth =np.arctan2(prey, prex)[keep]

    sig_bin = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=significance[keep])[0]
    counts =  np.histogram2d(prex[keep], prey[keep], bins=(bins, bins) )[0]
    sig_bin/=counts


    #sig_bin[counts==0]=None 
    #sig_bin = sig_bin 
    
    det_odds = np.zeros((len(bins), len(bins)))
    for ix in range(len(sig_bin)):
        
        for iy in range(len(sig_bin[ix])):
            
            det_odds[ix][iy] = np.nansum(sig_bin*photon_tensor[ix][iy] )


    det_odds/=np.max(det_odds)

    #sig_bin/=np.max(sig_bin)
    plt.pcolormesh(bins, binsy, det_odds.T, cmap='inferno') #, vmin=0, vmax=1)
    plt.xlabel("X [mm]", size=14)
    plt.ylabel("Y [mm]")
    cbar = plt.colorbar()
    cbar.set_label("Relative Det. Eff.")
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.show()


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
    #process("./data/0mg_hemi_2eV.dat", "0mG")
    #process("./data/0mg_lambertian.dat", "0mG")
    process("./data/0mG_lamb_1eV_fixv.dat", label="cone")
    #process("./data/500mG.dat", label="cone")
    #process("./data/0mg_highstat_2isheV.txt", "0mG")
    #process("./data/somefield_highstat_2isheV.txt", "~mG")
    #process("./data/old_comsol_petraj/test_z500.txt", "z100")
    #process("./data/0mG_down_lamb.dat", "0mG")
    #process("./data/0mG_down_lamb_1eV.dat", "0mG")
    #process("./data/500mG_down_cone.dat", "0mG")
    #process("./data/0mg_super_random_2isheV.dat", "500mG")
    process("./data/500mg_lambertian.dat", "0mG")
    
    #process("./data/500mg_hemi_2eV_newvolt.dat", "0mG")
    #process("./data/500mg_y_random.txt", "0mG")
