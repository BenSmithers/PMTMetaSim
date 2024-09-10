from scipy.interpolate import griddata, RectBivariateSpline
import numpy as np 
from math import log10,sqrt 


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