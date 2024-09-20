import uproot 
import numpy as np
from tqdm import tqdm 
import h5py as h5 
import os 
filename = "/Users/bsmithers/software/mPMT/build/output_365.root"
#filename = "./data/output_new.root"
wavelen = filename.split("_")[-1].split(".")[0]

data = uproot.open(filename)

init_x = []
init_y = []

final_x = []
final_y = []
final_z = []
ids = []
newname = "processed_optics_{}.hdf5".format(wavelen)
if os.path.exists(newname):
    os.remove(newname)
datafile = h5.File(newname, 'w')

initial_x = data["Photons_Master;1"]["PosX_Initial"].array()
initial_y = data["Photons_Master;1"]["PosY_Initial"].array()

for ip, photon_key in tqdm(enumerate(data.keys())):  
    if ip%200==0:
        print("collected {} photons".format(len(init_x)))
    if ip==0:
        continue
    photons = data[photon_key]["Photon_ID"].array()
    px =  data[photon_key]["PosX"].array()
    py =  data[photon_key]["PosY"].array()
    pz =  data[photon_key]["PosZ"].array()
    status = data[photon_key]["Step_Status"].array()
    unique_phot = np.unique(photons)

    mask = status==2

    final_x += px[mask].tolist()
    final_y += py[mask].tolist()
    final_z += pz[mask].tolist()

    ids += photons[mask].tolist()

    init_x  += (initial_x[ip-1]*np.ones_like(px[mask].tolist())).tolist()
    init_y  += (initial_y[ip-1]*np.ones_like(py[mask].tolist())).tolist()


all_data = {
    "init_x":init_x,
    "init_y":init_y,
    "final_x":final_x,
    "final_y":final_y,
    "final_z":final_z,
    "ids":ids
}


for key in all_data:
    datafile.create_dataset(key, data=all_data[key])
datafile.close()        

