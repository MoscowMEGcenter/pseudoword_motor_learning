import mne, os, sys, numpy as np
from configure import *
mne.set_log_level("ERROR")

#out_path = f'beta_{stim}clean_sc_{L_freq}_{H_freq}/'
out_path = f'beta_{stim}clean_{L_freq}_{H_freq}/'
data_path = os.path.join(os.getcwd(), "Sensor", "Evoked", out_path)
out_path = os.path.join(os.getcwd(), "Sensor", "Evoked_mean_chan", out_path)
os.makedirs(out_path, exist_ok = True)
sessions = [('active1', "st"), ('active2', "st"), ('active2', "end")]

for pref in sessions:
    print(pref)

    
    for subj in subjects:
        print(subj)
        beta_file = os.path.join(os.getcwd(), data_path, "{0}_{1}-{2}_{3}_beta-ave.fif".format(subj ,pref[0], pref[1], stim)) 
        beta_data = mne.Evoked(beta_file).pick_types("grad")

        temp = beta_data.copy().pick_types("planar1")
        temp.data += beta_data.copy().pick_types("planar2").data
        temp1 = temp.copy().pick_channels([temp.ch_names[0]])
        temp1.data[0, :] =  temp.data.mean(axis=0)
        temp1.times = beta_data.times - 1
        temp1.save(os.path.join(os.getcwd(), out_path, "{0}_{1}-{2}_{3}_merged_channels_beta-ave.fif".format(subj ,pref[0], pref[1], stim)) )
