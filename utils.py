from scipy.interpolate import griddata, RectBivariateSpline, interp1d
import os 
from math import pi
import numpy as np 
import json
from math import log10,sqrt 

from scipy.interpolate import interp2d, RegularGridInterpolator




class IrregularNDGridInterpolator:
    def __init__(self, 
                 points:np.ndarray,
                 values:np.ndarray,
                 method='linear',
                 fill_value=0, 
                 _skip=False):
        
        """
            points - (len(dimensions) by len(values))
            values - len(values)
            xi - (len(i) len(i+1) len(i+3)) for i dimensions

            Methods supported are “linear”, “nearest”, and “cubic”,
        """ 
        if not _skip:
            minvals = np.min(points, axis=1)
            maxvals = np.max(points, axis=1)
            print(minvals)
            print(maxvals)
            print("Generating Fine Values - ",len(minvals))
            self._fine_values = [
                np.linspace( minvals[i]*0.99, maxvals[i]*0.99, 10+i) for i in range(len(minvals))
            ]
            print("Building Mesh Grid")
            mesh_points = np.meshgrid(*self._fine_values)
            print("Mesh shape -",np.shape(mesh_points))

            print("Running Grid Eval")
            self._grid_eval = griddata(
                points=points.T, 
                values=values, 
                xi=mesh_points, 
                method=method
            )

            self._fill_value = fill_value

            if np.any(np.isnan(self._grid_eval)):
                print("Warning! Nans were found in the evaluation of griddata. Filling with {}".format(fill_value))
            self._grid_eval[np.isnan(self._grid_eval)] = fill_value
            self._method = method 
            self._data_int = RegularGridInterpolator(
                self._fine_values, 
                self._grid_eval, 
                fill_value=fill_value
            )           
    
    @classmethod
    def load_from_file(cls, filename):
        _obj = open(filename, 'rt') 
        data = json.load(_obj)
        _obj.close()
        new_one = IrregularNDGridInterpolator([], [], skip=True)

        new_one._data_int = RegularGridInterpolator(
            data["fine"], 
            data["eval"], 
            data["method"],
            data["fill"]
        )

        new_one._fine_values = np.array(data["fine"])
        new_one._grid_eval = np.array(data["eval"])
        new_one._method = data["method"]
        new_one._fill_value = data["fill"]
        return new_one

    def save_to_file(self, filename):
        out_data = {
            "fine": self._fine_values.tolist(),
            "eval": self._grid_eval.tolist(), 
            "method":self._method, 
            "fill":self._fill_value
        }

        _obj = open(filename, 'wt')
        json.dump(out_data, _obj, indent=4)
        _obj.close()


    def __call__(self, *values):
        return self._data_int(*values)

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
                    values:np.ndarray, 
                    linear_x = True, 
                    linear_y = True,
                    linear_values=True,
                    replace_nans_with= 0.0, 
                    interpolation='linear'):

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
            return self._data_int( ys, xs ,grid=grid)
        else:
            return 10**self._data_int( ys, xs ,grid=grid)
        
    
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

def get_loc(x:float, domain:list,closest=False):
    """
    Returns the indices of the entries in domain that border 'x' 
    Raises exception if x is outside the range of domain 
    Assumes 'domain' is sorted!! And this _only_ works if the domain is length 2 or above 
    This is made for finding bin numbers on a list of bin edges 
    """

    if len(domain)<=1:
        raise ValueError("get_loc function only works on domains of length>1. This is length {}".format(len(domain)))


    # I think this is a binary search
    min_abs = 0
    max_abs = len(domain)-1

    lower_bin = int(abs(max_abs-min_abs)/2)
    upper_bin = lower_bin+1

    while not (domain[lower_bin]<=x and domain[upper_bin]>=x):
        if abs(max_abs-min_abs)<=1:
            print("{} in {}".format(x, domain))
            raise Exception("Uh Oh")

        if x<domain[lower_bin]:
            max_abs = lower_bin
        if x>domain[upper_bin]:
            min_abs = upper_bin

        # now choose a new middle point for the upper and lower things
        lower_bin = min_abs + int(abs(max_abs-min_abs)/2)
        upper_bin = lower_bin + 1
    
    assert(x>=domain[lower_bin] and x<=domain[upper_bin])
    if closest:
        return( lower_bin if abs(domain[lower_bin]-x)<abs(domain[upper_bin]-x) else upper_bin )
    else:
        return(lower_bin, upper_bin)
    


def bilinear_interp(p0, p1, p2, q11, q12, q21, q22):
    """
    Performs a bilinear interpolation on a 2D surface
    Four values are provided (the qs) relating to the values at the vertices of a square in the (x,y) domain
        p0  - point at which we want a value (len-2 tuple)
        p1  - coordinates bottom-left corner (1,1) of the square in the (x,y) domain (len-2 tuple)
        p2  - upper-right corner (2,2) of the square in the (X,y) domain (len-2 tuple)
        qs  - values at the vertices of the square (See diagram), any value supporting +/-/*
                    right now: floats, ints, np.ndarrays 
        (1,2)----(2,2)
          |        |
          |        |
        (1,1)----(2,1)
    """

    
    # check this out for the math
    # https://en.wikipedia.org/wiki/Bilinear_interpolation

    x0 = p0[0]
    x1 = p1[0]
    x2 = p2[0]
    y0 = p0[1]
    y1 = p1[1]
    y2 = p2[1]

    if not (x0>=x1 and x0<=x2):
        raise ValueError("You're doing it wrong. x0 should be between {} and {}, got {}".format(x1,x2,x0))
    if not (y0>=y1 and y0<=y2):
        raise ValueError("You're doing it wrong. y0 should be between {} and {}, got {}".format(y1,y2,y0))

    # this is some matrix multiplication. See the above link for details
    # it's not magic, it's math. Mathemagic 
    mat_mult_1 = [q11*(y2-y0) + q12*(y0-y1) , q21*(y2-y0) + q22*(y0-y1)]
    mat_mult_final = (x2-x0)*mat_mult_1[0] + (x0-x1)*mat_mult_1[1]

    return( mat_mult_final/((x2-x1)*(y2-y1)) )

class SpecialInterpolator:
    def __init__(self, 
                 points:np.ndarray,
                 values:np.ndarray,
                 method='linear',
                 fill_value=0):
        """`
            We build an interpolator for each theta/phi combination

            For a given point, we evaluate the (x,y) interpolator for each corner on the (theta,phi)
                grid around the point. 
                Then do a bilinear interpolation of those interpolators 
        """
        from copy import copy 
        all_zen = points[2] 
        all_azi = points[3]
        self.zens= np.unique(points[2])
        self.azis = np.unique(points[3])
        self.fill_value = fill_value

        self.interpolators = {}

        for iz, zen in enumerate(self.zens):
            self.interpolators[iz] = {}
            for ia, azi in enumerate(self.azis):
                mask = np.logical_and(zen==all_zen, azi==all_azi)
                these_x = points[0][mask]
                these_y = points[1][mask]

                self.interpolators[iz][ia] = Irregular2DInterpolator(
                    these_x, these_y, values[mask]
                )
    def __call__(self, x, y, zen, azi):
        # now we do that bilinear interpolator
        if zen<np.min(self.zens) or zen>np.max(self.zens) or azi<np.min(self.azis) or azi>np.max(self.azis):
            return self.fill_value
        
        z_low, z_up = get_loc(zen, self.zens)
        a_low, a_up = get_loc(azi, self.azis)

        return bilinear_interp(
            (zen, azi),
            (self.zens[z_low], self.azis[a_low]),
            (self.zens[z_up], self.azis[a_up]),
            self.interpolators[z_low][a_low](x, y),
            self.interpolators[z_low][a_up](x, y),
            self.interpolators[z_up][a_low](x, y),
            self.interpolators[z_up][a_up](x, y)
        )