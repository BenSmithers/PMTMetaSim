from scipy.interpolate import griddata, RectBivariateSpline, interp1d
import os 
from math import pi
import numpy as np 
from math import log10,sqrt 

from scipy.interpolate import interp2d


def load_newest_dyn():
    data = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "data","dynode_tall.txt"),
        delimiter=",",
        comments="%"
    ).T 
    
    thetas = data[0]*pi/180
    phis = data[1]*pi/180
    odds = (data[3]*1 + data[4]*0.1 + data[5]*0.01)/(data[3]+data[4]+data[5])

    odds[(data[3]+data[4]+data[5])==0] = 0.0


    #odds = (0.5+odds)/2

    i2d = Irregular2DInterpolator(
        thetas, phis, odds
    )
    return i2d


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
    odds = (data[2] + data[3]*0.01 + data[4]*0.01)/(data[2]+data[3]+data[4])


    i2d = Irregular2DInterpolator(
        thetas, phis, odds
    )
    return i2d


class Irregular2DInterpolator:
    """
        This is used to make a 2D interpolator given a set of data that do not lie perfectly on a grid.
        This is done using scipy griddata and scipy RectBivariateSpline 
        interpolation can be `linear` or `cubic` 
        if linear_x/y, then the interpolation is done in linear space. Otherwise, it's done in log space
            setting this to False is helpful if your x/y values span many orders of magnitude 
        if linear_values, then the values are calculated in linear space. Otherwise they'll be evaluated in log space- but returned in linear space 
            setting this to False is helpful if your data values span many orders of magnitude 
        By default, nans are replaced with zeros. 
    """
    def __init__(self, xdata:np.ndarray, 
                 ydata:np.ndarray,
                   values:np.ndarray, linear_x = True, linear_y = True, linear_values=True,
                   replace_nans_with= 0.0, interpolation='linear'):

        self._nomesh_x = xdata
        self._nomesh_y = ydata 
        self._values = values if linear_values else np.log10(values)
        self._linear_values = linear_values
        if linear_x:
            self._xfine = np.linspace(min(self._nomesh_x), 
                                      max(self._nomesh_x), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)
        else:
            self._xfine = np.logspace(log10(min(self._nomesh_x)), 
                                      log10(max(self._nomesh_x)), 
                                      int(sqrt(len(self._nomesh_x)))*2, endpoint=True)

        
        if linear_y:
            self._yfine = np.linspace(min(self._nomesh_y), 
                                      max(self._nomesh_y), 
                                      int(sqrt(len(self._nomesh_y)))*2+1, endpoint=True)
        else:
            self._yfine = np.logspace(log10(min(self._nomesh_y)), 
                                      log10(max(self._nomesh_y)), 
                                      int(sqrt(len(self._nomesh_y)))*2+1, endpoint=True)


        mesh_x, mesh_y = np.meshgrid(self._xfine, self._yfine)

        # usee grideval to evaluate a grid of points 
        grid_eval = griddata(
            points=np.transpose([self._nomesh_x, self._nomesh_y]),
            values=self._values, 
            xi=(mesh_x, mesh_y),
            method=interpolation
        )
        
        # if there are any nans, scipy 
        if np.any(np.isnan(grid_eval)):
            print("Warning! Nans were found in the evaluation of griddata - we're replacing those with zeros")
        grid_eval[np.isnan(grid_eval)] = replace_nans_with

        # and then prepare an interpolator 
        self._data_int = RectBivariateSpline(
            self._xfine, 
            self._yfine, 
            grid_eval.T
        )

    def __call__(self, xs, ys, grid=False):
        if self._linear_values:
            return self._data_int( xs, ys ,grid=grid)
        else:
            return 10**self._data_int( xs, ys ,grid=grid)
        
    
def perlin(granularity,octave=5)->np.ndarray:
    """
    returns a mesh of perlin noise given a seed and granularity 
    
    returns numpy array with values ranging between -0.5 and 0.5
    """

    lin = np.linspace(0,octave,200,endpoint=False)
    x,y = np.meshgrid(lin, lin)

    # permutation table
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # _fade factors
    u = _fade(xf)
    v = _fade(yf)
    # noise components
    n00 = _gradient(p[p[xi]+yi],xf,yf)
    n01 = _gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = _gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = _gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = _lerp(n00,n10,u)
    x2 = _lerp(n01,n11,u) # FIX1: I was using n10 instead of n01

    values = np.zeros(shape=(granularity, granularity))
    xs = np.array(range(200))*granularity/200
    evals = interp2d(xs,xs,_lerp(x1,x2,v))
    values = evals(range(granularity),range(granularity))

    return values # FIX2: I also had to reverse x1 and x2 here

def _lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def _fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def _gradient(h,x,y):
    "grad converts h to the right _gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y