import os 
import numpy as np 

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from utils import Irregular2DInterpolator

def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    
    FROM: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def load_file(filename):

    obj = open(filename, 'rt')

    this_data = []
    working_line = []
    n_col = 14
    for i, line in enumerate(obj.readlines()):
        if False : # i==0:
            # this is the header 
            n_col = len(line.split(","))
            print(n_col)
        else:
            extract = [float(x) for x in list(filter( lambda x:len(x)!=0, line.split(" ")))]
            extract = extract[1:]
            working_line += extract
            this_data.append(extract)

    print(np.shape(this_data))            
    return np.array(this_data).T

def process(filename, label):
    test = load_file(filename)

    prex = test[0]
    prey = test[1]
    prez = test[2]

    xpos = test[6]
    ypos = test[7]
    zpos = test[8]

    vx = test[9]
    vy = test[10]
    vz = test[11]

    norm_plane = np.sqrt(vx**2 + vy**2)

    # >1 is down-going
    # <1 is horizontal
    color = np.arctan(vz/norm_plane)
    print("{} - {}".format(np.min(color), np.max(color)))
    colors = get_color(color,3.14159/2, "viridis")

    norm = -1*np.sqrt(vx**2 + vy**2 + vz**2)*50
    vx/=norm
    vy/=norm 
    vz/=norm 

    renorm = np.abs(norm/50)

    print("{} - {}".format(min(renorm), max(renorm)))
    vbin = np.logspace(7, np.log10(2e7), 100)
    histo = np.histogram(renorm[np.logical_not(np.isnan(renorm))], bins=vbin)
    plt.stairs(histo[0],vbin)
    plt.yscale('log')
    plt.xlabel("Velocity [m/s]", size=14)
    plt.ylabel("Counts", size=14)
    plt.xscale('log')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.scatter(prex, prey, prez)
    ax.quiver(xpos, ypos, zpos, vx, vy,vz, alpha=0.3, colors=colors)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_zlim([-0.2, 0])
    
    plt.savefig("./plots/{}.png".format(label), dpi=400)
    plt.show()
    plt.close()

    

if __name__=="__main__":
    process("./data/old_comsol_petraj/pe_trajectory_0mg.txt", "0mG")
    process("./data/old_comsol_petraj/pe_trajectory_250mg.txt", "250mG")
    process("./data/old_comsol_petraj/pe_trajectory_500mg.txt", "500mG")
    