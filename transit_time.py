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

from is_hit import new_dyn_odds, prepare_photonics

g4bins  = np.arange(-0.305,0.305,0.01)

stepsize = g4bins[1]-g4bins[0]
bins = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
binsy = np.linspace(g4bins[0] - 0.5*stepsize, g4bins[-1]+0.5*stepsize, len(g4bins)+1)
pmt_costheta =  np.linspace(0, 1, 30)
pmt_azimuth = np.linspace(-pi, pi, 31)

pmt_a = 0.254564663
pmt_b = 0.254110205
pmt_c = 0.186002389

photon_tensor = prepare_photonics(True)


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

    stop_time = test[-1]*1e9
    
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



    to_bin = significance[keep]*stop_time[keep]

    sig_bin = np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=to_bin)[0]
    counts =  np.histogram2d(prex[keep], prey[keep], bins=(bins, bins), weights=significance[keep])[0]
    sig_bin/=counts

    
    det_odds = np.zeros((len(bins), len(bins)))
    for ix in range(len(sig_bin)):
        for iy in range(len(sig_bin[ix])):
            det_odds[ix][iy] = np.nansum(sig_bin*photon_tensor[ix][iy] )


    
    #sig_bin/=np.max(sig_bin)
    plt.pcolormesh(bins, binsy, det_odds.T, cmap='inferno') #, vmin=0, vmax=1)
    plt.xlabel("X [mm]", size=14)
    plt.ylabel("Y [mm]")
    cbar = plt.colorbar()
    cbar.set_label("Relative Det. Eff.")
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.show()

if __name__=="__main__":
    process("./data/0mg_lamb_checktime.dat", "0mG")
