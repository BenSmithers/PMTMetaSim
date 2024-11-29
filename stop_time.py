from otherdata import load_file
import sys 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
import numpy as np 
from math import log, sqrt 

conv = 2*sqrt(2*log(2))

if __name__=="__main__":

    test = load_file(sys.argv[1])

    times = np.array(test[-1])*(1e9)

    bins = np.linspace(2, 100, 100)
    centers = (bins[1:] + bins[:-1])*0.5

    data = np.histogram(times, bins)[0]
    data = data/np.sum(data)


    def gaus(xs, params):
        return params[0]*np.exp(-0.5*((xs-params[1])/params[2])**2)


    def metric(params):
        evaluated = gaus(centers, params) 
        return np.sum((evaluated - data)**2 )
    
    x0 = [0.2, 65, 10]

    res = minimize(metric, x0=x0).x

    plt.stairs(data, bins)
    plt.text(20, 0.1, r"$\mu$="+"{:.2f}".format(res[1]))
    plt.text(20, 0.11, r"FWHM="+"{:.2f}".format(conv*res[2]))
    plt.xlabel("Dynode Hit Time [ns]", size=14)
    plt.ylabel("Arb")    
    plt.show()
