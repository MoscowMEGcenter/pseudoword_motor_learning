import mne
import numpy as np
from configure import *
import matplotlib.pyplot as plt
import pickle

folder = f"beta_{stim}clean_epoch_{L_freq}_{H_freq}"
out_path = os.path.join(os.getcwd(), 'Sensor', 'Rastr_ord_ave', folder)

dictionary = dict()

for pref in session:
    data_path = os.path.join(os.getcwd(), 'Sensor', 'TFR', folder)
    os.makedirs(out_path, exist_ok = True)

    with open(os.path.join(os.getcwd(), "Sensor", "TFR", data_path, "{0}_{1}-{2}_{3}_resp_time.txt".format(subjects[0], pref[0], pref[1], stim)), "r") as f:
        times = list(map(int, f.read().split("\n")))
    new_times = np.zeros_like(np.array(times))

    freq_file = os.path.join(os.getcwd(), data_path, "{0}_{1}-{2}_{3}_int_50ms-tfr.h5".format(subjects[0],pref[0], pref[1], stim))   
    ave_epochs_data = np.zeros_like(mne.time_frequency.read_tfrs(freq_file)[0].data)

    out_pic_path = os.path.join(os.getcwd(), 'Sensor', 'Rastr_ord_ave_', folder, "_".join(pref))
    os.makedirs(out_pic_path, exist_ok = True)
    
    
    for subj in subjects:
        print("\t", subj)
        freq_file = os.path.join(os.getcwd(), data_path, "{0}_{1}-{2}_{3}_int_50ms-tfr.h5".format(subj,pref[0], pref[1], stim))   
        freq_data = mne.time_frequency.read_tfrs(freq_file)[0]
        with open(os.path.join(os.getcwd(), "Sensor", "TFR", data_path, "{0}_{1}-{2}_{3}_resp_time.txt".format(subj, pref[0], pref[1], stim)), "r") as f:
            times = np.array(list(map(int, f.read().split("\n"))))

        ranks = times.argsort()

        new_times += times[ranks]
        ave_epochs_data += freq_data.data[ranks,:,:,:]
        if subj == subjects[0]:
            a, _, _, d = freq_data.data[ranks,:,:,:].shape
            epo_d = (freq_data.data[ranks,0::3,:,:] + freq_data.data[ranks,1::3,:,:]).sum(axis=1).reshape((1, a, d))/204
        else:
            epo_d = np.concatenate((epo_d, (freq_data.data[ranks,0::3,:,:] + freq_data.data[ranks,1::3,:,:]).sum(axis=1).reshape((1, a, d))))/204
            


    
    dictionary["{0}_{1}".format(pref[0], pref[1])] = epo_d
    print(epo_d.shape, epo_d.min(), epo_d.mean(), epo_d.max())
    ave_epochs_data = ave_epochs_data / len(subjects)
    temp = np.zeros_like(ave_epochs_data)

    temp[:,0,:,:] = ave_epochs_data.mean(axis=1)
    ave_epochs_data = temp
    new_times = new_times / len(subjects)
    a, b, _, c = ave_epochs_data.shape
        
    epoch_dat = mne.EpochsArray(ave_epochs_data.reshape((a, b, c)), freq_data.info, tmin=freq_data.times[0])
     
    epoch_dat = epoch_dat.crop(tmin=-1.8, tmax=1.8)
    epoch_dat = epoch_dat.pick_types("grad")
    channels = epoch_dat.ch_names
    time_to_plot = -np.array(new_times)/1000
    dictionary["{0}_{1}_react".format(pref[0], pref[1])] = time_to_plot 
    for ch in range(0, 204, 2):
        plt.figure()

        #epoch_dat.plot_image(picks=[channels[ch], channels[ch + 1]], order=ranks, combine="mean", scalings=dict(grad=1), units=dict(grad="Au"), vmin=-1, vmax=1, show=False, overlay_times=time_to_plot)[0]
        epoch_dat.plot_image(picks=[channels[ch], channels[ch + 1]], combine="mean", scalings=dict(grad=1), units=dict(grad="Au"), vmin=-0.1, vmax=0.1, show=False, overlay_times=time_to_plot)[0]
            
        plt.savefig(os.path.join(out_pic_path, "+".join([channels[ch], channels[ch + 1][3:]]) + ".png"))
dictionary["times"] = freq_data.times
with open("rastr.pickle", "wb") as f:
    pickle.dump(dictionary, f)
