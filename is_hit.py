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
pmt_costheta =  np.linspace(0, pi, 50)
pmt_azimuth = np.linspace(-pi, pi, 51)

pmt_a = 0.254564663
pmt_b = 0.254110205
pmt_c = 0.186002389

def prepare_photonics(norm = False, wavelen=365):
    """
        Prepares and returns a PE-production tensor for particles generated at a given location 
    """
    _filename = "./processed_optics_new.hdf5".format(wavelen)
    N_PHOT = 10000

    data = h5.File(_filename, 'r')

    initial_x = np.array(data["init_x"])/1000
    initial_y =  np.array(data["init_y"])/1000
    hit_x =  np.array(data["final_x"])/1000
    hit_y =  np.array(data["final_y"])/1000
    hit_z =  np.array(data["final_z"])/1000

    czen = (pi/2) -np.arctan(hit_z/np.sqrt(hit_x**2 + hit_y**2))
    aziumuth = np.arctan2(hit_y, hit_x)
    print("{} - {}".format(min(czen), max(czen)))

    # histogram all of the data 
    photons = np.histogramdd(
        sample = (
            initial_x, initial_y, czen, aziumuth
        ),
        bins = (
            bins, binsy,pmt_costheta, pmt_azimuth
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
                plt.pcolormesh(bins, binsy,np.log10(photons[ix][iy]+1).T)
                plt.title("{} - {}".format(ix, iy))
                plt.xlabel(r"$\theta_{hit}$ [rad]")
                plt.ylabel("$\phi_{hit}$ [rad]")
                plt.colorbar()
                plt.show()
            if norm:
                prenorm[ix][iy]/=np.sum(prenorm[ix][iy])

    return prenorm

_photon_tensor = prepare_photonics(False)

def load_dyn_odds():
    data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "data","second_gen_data","dyn_sim_geo.dat"),
        delimiter=",",
        comments="#"
    ).T 

    thetas = -1*np.arctan(data[1]/data[0])
    odds = (data[2] + data[3]*0.1+ data[4]*0.01)/(data[2]+data[3]+data[4]) # odds of hitting first dynode

    return interp1d( thetas, odds )

def new_dyn_odds():
    data = np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data","dyn_new.dat"),
            delimiter=",",
            comments="#"
        ).T

    thetas = data[0]*pi/180
    phis = data[1]*pi/180
    odds = (data[2] + data[3]*0.01 + data[4]*0.00)/(data[2]+data[3]+data[4])


    i2d = Irregular2DInterpolator(
        thetas, phis, odds
    )
    return i2d

def process(filename, label, reload_photonics = -1):
    if reload_photonics <0:
        photon_tensor = _photon_tensor  # get teh 
    else:
        photon_tensor = prepare_photonics(False, reload_photonics)

    test = load_file(filename) 

    interpo = new_dyn_odds()
    interpo1d = load_dyn_odds()

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

    spine_penalty = np.abs(xpos) / 0.004
    #spine_penalty = 1/spine_penalty
    spine_penalty[spine_penalty>1] = 1.2
    #spine_penalty += 0.5 
    #spine_penalty/=1.5


    dyn_x = -1*np.ones_like(xpos)
    dyn_x[xpos<0]*=-1 

    significance = np.ones_like(vx)
    should_eval = np.ones_like(dyn_x).astype(bool)


    these_thetas = np.arctan(dyn_x*vx[should_eval]/vz[should_eval])
    these_phis = np.abs(np.arctan(vx/vy))
    print( "{} - {}".format(np.min(these_phis) , np.max(these_phis)))
    
    #these_odds = interpo(these_thetas, these_phis, grid=False)
    #these_odds = interpo1d(these_thetas) #*np.cos(these_phis)
    these_odds = np.ones_like(these_thetas)

    # trying out a stupid-simple model
    these_odds -= np.sin(these_thetas)*np.sin(these_phis)
    these_odds[these_thetas < 0] = 1.0
    these_odds /= np.max(these_odds)

    significance[should_eval] = these_odds #*spine_penalty

    significance[bad] = 0.0

    pzenith = (pi/2)-np.arctan(prez/np.sqrt(prex**2 + prey**2))
    pazimuth =np.arctan2(prey, prex)

    sig_bin = np.histogram2d(pzenith, pazimuth, bins=(pmt_costheta, pmt_azimuth), weights=significance)[0]
    counts =  np.histogram2d(pzenith, pazimuth, bins=(pmt_costheta, pmt_azimuth), weights=np.ones_like(significance))[0]
    #sig_bin = np.histogram2d(prex, prey, bins=(bins, binsy), weights=significance)[0]
    #counts =  np.histogram2d(prex, prey, bins=(bins, binsy) )[0]
    sig_bin/=counts


    #sig_bin[counts==0]=None 
    #sig_bin = sig_bin 
     
    det_odds = np.zeros((len(bins), len(bins)))
    for ix in range(len(photon_tensor)):
        for iy in range(len(photon_tensor[ix])):
            det_odds[ix][iy] = np.nansum(sig_bin*photon_tensor[ix][iy] )


    det_odds/=np.max(det_odds)

    #sig_bin/=np.max(sig_bin)
    plt.clf()
    plt.pcolormesh(bins, binsy, sig_bin.T, cmap='inferno') #, vmin=0, vmax=1)
    #plt.pcolormesh(bins, binsy, 0-sig_bin.T, cmap='RdBu') #, vmin=0, vmax=1)
    plt.xlabel("X [m]", size=14)
    plt.ylabel("Y [m]")
    cbar = plt.colorbar()
    cbar.set_label("Relative Det. Eff.")
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.savefig("./plots/det_eff_{}.png".format(label), dpi=400)
    #plt.show()

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
    #process("./data/0mg_hemi_2eV.dat", "0mG")
    #first = process("./data/0mG_slowpes.dat", "0mG_365nm", 365)
    #second = process("./data/0mG_slowpes.dat", "0mG_405nm", 405)
    #third = process("./data/0mG_slowpes.dat", "0mG_540nm", 540)

    #print(np.nanmean(first/second))
    #print(np.nanmean(first/third))
    #process("./data/0mG_lamb_1eV_fixv.dat", label="cone")
    #process("./data/500mG.dat", label="cone")
    #process("./data/0mg_highstat_2isheV.txt", "0mG")
    #process("./data/somefield_highstat_2isheV.txt", "~mG")
    #process("./data/old_comsol_petraj/test_z500.txt", "z100")
    #process("./data/0mG_down_lamb.dat", "0mG")
    #process("./data/0mG_down_lamb_1eV.dat", "0mG")
    process("./data/0mG_latest.dat", "0mG")
    process("./data/250mG_z_latest.dat", "250mG_z")
    process("./data/250mG_y_latest.dat", "250mG_y")
    #process("./data/second_gen_data/250y_highstat_2isheV.txt", "250mG")
    #process("./data/old_comsol_petraj/test_z500.txt", "500mG_z")
    #process("./data/0mg_super_random_2isheV.dat", "500mG")
    #process("./data/second_gen_data/0mg_lamb_checktime.dat", "0mG")
    #process("./data/0mg_fieldgrad.dat", "test")
    #process("./data/0mg_fieldgrad_more.dat", "test")
    #process("./data/0mg_fieldgrad_250strong.dat", "test")
    #process("./data/0mG_diffV_1eV_cone.dat", "0mG")
    #process("./data/500mg_y_random.txt", "0mG")
    #process("./data/0mG_highV_1eV_cone.dat", "")
