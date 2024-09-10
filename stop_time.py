from otherdata import load_file
import sys 
import matplotlib.pyplot as plt 
import numpy as np 

if __name__=="__main__":

    test = load_file(sys.argv[1])

    times = np.array(test[-1])*(1e9)

    bins = np.linspace(2, 100, 100)

    data = np.histogram(times, bins)[0]
    data = data/np.sum(data)

    plt.stairs(data, bins)
    plt.xlabel("Stop Time [ns]", size=14)
    plt.ylabel("Arb")    
    plt.show()
