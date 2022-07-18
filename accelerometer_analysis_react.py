import mne, os
import numpy as np
from mne import set_config
import matplotlib.pyplot as plt
from configure import *
import pickle

def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b[:-1])

def meadian(data, window = 60):
    temp = np.zeros_like(data)
    temp[:window] = data[:window]
    for i in range(window, data.shape[0]):
        temp[i - window //2] = np.median(data[i-window:i])
    return temp 

def rms(data):
    return np.power(np.mean(np.power(data, 2)), 0.5)
def anti_vibros(data, window = 6, eps = 0.001):
    difference = [0]
    k = 0
    df = []
    for i in range(1, data.shape[0]):
        df.append(abs((data[i] - data[i - 1])))
     
        if abs((data[i] - data[i - 1]) - rms(difference)) > eps:
            data[i] = data[i - 1] + rms(difference)*np.sign(data[i] - data[i - 1])
            k += 1
        difference.append(data[i] - data[i - 1])
        if len(difference) > window:
            difference.pop(0)

    return data

       

mne.set_log_level("ERROR")

init()
#session = [session[0]]
stimulus = ["react"]
for pref in session:
    for stim in stimulus:
        massive_left = 0
        massive_right = 0
        for ind, subj in enumerate(subjects):

                
            print(subj)

            data_path = '/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/RAW_trans/'
            out_path = f'accel_{stim}clean_wrong/'
            os.makedirs(os.path.join(os.getcwd(), "Accel", out_path), exist_ok = True)
            mne.set_log_level("ERROR")
            raw_file = os.path.join(data_path, subj,"{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0]))
            try:
                raw_data = mne.io.Raw(raw_file, preload=True)
            except:
                print("\t", "File wasn't found. Please, check the filename format")
            print("Raw ready")
            events_right = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], "r_right_hand")))
            events_left = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], "r_left_hand")))
            #events = np.concatenate((events_right, events_left))
            #print(events.shape)
            print("Events ready")
            try:
                epochs_right = mne.Epochs(raw_data, events_left, event_id = None, tmin = -2.0, tmax = 2.0, preload = True)
                epochs_left = mne.Epochs(raw_data, events_right, event_id = None, tmin = -2.0, tmax = 2.0, preload = True) 
            except:
                print("Bad in", subj, pref)
                raise Exception("I belive I can fly!!!")
            chm = [temp for temp in epochs_left.ch_names if "MISC" in temp]
            start = epochs_left.ch_names.index(chm[0])
            
            sqr = lambda x: np.power(x, 2)
            data_right = np.mean(epochs_right.get_data(), axis=0)
            data_left = np.mean(epochs_left.get_data(), axis=0)
            #t = np.concatenate((data_right, data_left))
            #t = np.mean(t, axis=0)
            print(data_right.shape)
            print(data_left.shape)
            data_right -= np.mean(data_right, axis=1)[:, np.newaxis]
            data_left -= np.mean(data_left, axis=1)[:, np.newaxis]
            fig = plt.figure()
            plt.ylim(-0.02, 0.1)
            ax = plt.subplot(111)
            
            data1 = np.sqrt(sqr(anti_vibros(data_left[start])) + sqr(anti_vibros(data_left[start + 1])) + sqr(anti_vibros(data_left[start + 2])))
            #data1 = np.sqrt(sqr((t[start])) + sqr((t[start + 1])) + sqr((t[start + 2])))

  


            data1 -= np.mean(data1[:300])
            ax.plot(epochs_left.times[:-30], data1[:-30], label="LEFT")

            data2 = np.sqrt(sqr(anti_vibros(data_right[start + 3])) + sqr(anti_vibros(data_right[start + 4])) + sqr(anti_vibros(data_right[start + 5])))


            data2 -= np.mean(data2[:300])
            ax.plot(epochs_right.times[:-30], data2[:-30], label="RIGHT")

            fig.savefig(os.path.join(os.getcwd(), "Accel", out_path, f"{subj}_{stim}_{pref[0]}-{pref[1]}_all.png"))
            plt.close()
            if ind == 0:
                massive_left = np.reshape(data1, (1, data1.shape[0]))
                massive_right = np.reshape(data2, (1, data2.shape[0]))
            else:
                massive_left = np.concatenate((massive_left, np.reshape(data1, (1, data1.shape[0]))))
                massive_right = np.concatenate((massive_right, np.reshape(data2, (1, data2.shape[0]))))
        fig = plt.figure()
        plt.ylim(-0.02, 0.1)
        ax = plt.subplot(111)
        massive_left = np.mean(massive_left, axis = 0) 
        massive_right = np.mean(massive_right, axis = 0)
        ax.plot(epochs_left.times[:-30], massive_left[:-30], label="LEFT")
        ax.plot(epochs_right.times[:-30], massive_right[:-30], label="RIGHT")
        fig.savefig(os.path.join(os.getcwd(), "Accel", out_path, f"all_subj_{stim}_{pref[0]}-{pref[1]}_all.png"))
        plt.close()

