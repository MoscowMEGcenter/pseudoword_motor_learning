#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:51:02 2021

@author: niherus
"""

import mne
import numpy as np
from configure import *
import os
import matplotlib.pyplot as plt
mne.set_log_level("ERROR")

folder = "rastr"
path = "/net/server/data/home/niherus/Desktop/Python/Make_source_for_beta/MAIN_SCRIPT/Sensor/Rastr_ogib_files_new/beta_reactclean_epoch_15_26/"
ppath = '/net/server/data/home/niherus/Desktop/Python/Make_source_for_beta/MAIN_SCRIPT/Sensor/TFR/beta_reactclean_epoch_15_26'
f_format = "{0}_{1}_react_beta-epo.fif"
alpha = 0.4
get = path + f_format
test = mne.read_epochs(get.format(subjects[0], "-".join(session[0])))
channels = test.ch_names
l, w, h = test.get_data().shape

timus = np.zeros((27, len(subjects)))
for pref in session:
    arr = np.zeros((len(subjects), l, w, h))
    out_pic_path = os.path.join(os.getcwd(), 'Sensor', 'Rastr_ogib_ord_new', folder, "_".join(pref))
    os.makedirs(out_pic_path, exist_ok=True)
    print(pref, "started")
    for ind, subj in enumerate(subjects):
        with open(os.path.join(ppath, "{0}_{1}-{2}_{3}_resp_time.txt".format(subj, pref[0], pref[1], stim)), "r") as f:
            times = list(map(int, f.read().split("\n")))
            times.sort()
        time_to_plot = np.array(times)/1000
        timus[:, ind] = -time_to_plot#.sort()
        print("\t", ind + 1, subj)
        arr[ind] = mne.read_epochs(get.format(subj, "-".join(pref))).get_data()
    time_to_plot = timus.mean(axis=1)
    data = arr.mean(axis=0)
    new_data = np.zeros_like(data)
    new_data[0] = data[0]
    for i in range(1, data.shape[0]):
        new_data[i] = alpha * data[i] + (1 - alpha) * new_data[i - 1]
    epoch_dat = mne.EpochsArray(new_data, test.info, tmin=test.times[0])
    epoch_dat = epoch_dat.pick_types("grad")
    for ch in range(0, 204, 2):
        epoch_dat.plot_image(picks=[channels[ch], channels[ch + 1]], combine="mean", scalings=dict(grad=1), units=dict(grad="Au"), vmin=-0.5, vmax=0.5, show=False, overlay_times=time_to_plot)[0]
     
       
        plt.savefig(os.path.join(out_pic_path, "+".join([channels[ch], channels[ch + 1].replace("MEG", "")])  + ".png"))
        plt.close()
    
