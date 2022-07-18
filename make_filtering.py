import mne, os
import numpy as np
from mne import set_config
from joblib import Parallel, delayed
from configure import *
import pickle


set_config("MNE_MEMMAP_MIN_SIZE", "1M")
set_config("MNE_CACHE_DIR", ".tmp")

mne.set_log_level("ERROR")
os.makedirs("baseline", exist_ok=True)
init()

def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b)

data_path = '/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/RAW_trans/'
out_path = f'beta_{stim + "NEW"}_{L_freq}_{H_freq}/'
os.makedirs(os.path.join(os.getcwd(), "Sensor", "TFR", out_path), exist_ok = True)

with open("configure.py", "r") as f_in:
    settings = f_in.readlines()
    with open(os.path.join(os.getcwd(), "Sensor", "TFR", out_path, "confige.log"), "w") as f_out:
        f_out.writelines(settings)

with open("check_channels.txt", "r") as f_ch:
    ch_names = list(map(str.strip, f_ch.readlines()))
    


def calculate_beta(subj ):
    print(subj)
    mne.set_log_level("ERROR")
    freqs = np.arange(L_freq, H_freq, f_step)
    raw_file = os.path.join(data_path, subj,"{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0]))
    try:
        raw_data = mne.io.Raw(raw_file, preload=True)
    except:
        print("\t", "File wasn't found. Please, check the filename format")
        return 0
    #print(raw_data.info)
    #import sys
    #sys.exit()
    #raw_data = raw_data.pick(picks=ch_names + ["STI101"])
    raw_data = raw_data.filter(l_freq=70, h_freq=None)
    raw_data.save(os.path.join("/net/server/mnt/home/inside/filtered", "{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0])), overwrite=True)
    
    print(subj, pref, " ended!")

p_func = delayed(calculate_beta)
parallel = Parallel(6, max_nbytes = None)

answ = input(f"Are you sure? Your beta data in {out_path} will be rewrited.\n")
if  answ == "" or answ != "yes":
    print("Analisys canceled")
else:
    parallel(p_func(subject) for subject in subjects)

