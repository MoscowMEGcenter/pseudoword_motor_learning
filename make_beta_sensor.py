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
'''
def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b)
'''

def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b[:-1])

data_path = '/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/RAW_trans/'
out_path = f'beta_{stim}clean_otr_sub_{L_freq}_{H_freq}/'
os.makedirs(os.path.join(os.getcwd(), "Sensor", "TFR", out_path), exist_ok = True)

with open("configure.py", "r") as f_in:
    settings = f_in.readlines()
    with open(os.path.join(os.getcwd(), "Sensor", "TFR", out_path, "confige.log"), "w") as f_out:
        f_out.writelines(settings)
         

def calculate_beta(subj ):
    print(subj)
    for pref in session:
        mne.set_log_level("ERROR")
        freqs = np.arange(L_freq, H_freq, f_step)
        raw_file = os.path.join(data_path, subj,"{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0]))
        try:
            raw_data = mne.io.Raw(raw_file, preload=True)
        except:
            print("\t", "File wasn't found. Please, check the filename format")
            return 0
        

        picks = mne.pick_types(raw_data.info, meg = True, eog = True)
        
        
        if stim == "react":
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], "w", pref[1])))
            events_react = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], stim[0], pref[1])))
            
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, picks = picks, preload = True)
            except:
                print("Bad in", subj, pref, "stim")
                raise Exception("I belive I can fly!!!")
            epochs.resample(300)

            freq_show_baseline = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False).crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            
            b_line  = freq_show_baseline.data.mean(axis=-1)
            del freq_show_baseline
            try:
                epochs = mne.Epochs(raw_data, events_react, event_id = None, tmin = period_start, tmax = period_end, picks = picks, preload = True)
            except:
                print("Bad in", subj, pref, "react")
                raise Exception("I belive I can fly!!!")            
            epochs.resample(300)
            del raw_data
            freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False)
            del epochs, freqs

            temp = freq_show.data.sum(axis=1)
            b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)

            freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])
            freq_show.data = 10 * np.log10(freq_show.data/b_line[:, np.newaxis])
            freq_show.freqs = np.array([22.5])
        elif "hand" in stim or "foot" in stim:
            if "r" == stim[0]:
                events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], "w" + stim[1:])))

           
                events_react = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], stim)))
                
                try:
                    epochs = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 0.0, picks = picks, preload = True)
                except:
                    print("Bad in", subj, pref, "stim")
                    raise Exception("I belive I can fly!!!")
                epochs.resample(300)

                freq_show_baseline = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False).copy().crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
                b_line  = freq_show_baseline.data.mean(axis=-1)
                

                del freq_show_baseline
                
                try:
                    epochs = mne.Epochs(raw_data, events_react, event_id = None, tmin = period_start, tmax = period_end, picks = picks, preload = True)
                except:
                    print("Bad in", subj, pref, "react")
                    raise Exception("I belive I can fly!!!")            
                epochs.resample(300)
                del raw_data
                freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False)

                #b_line  = freq_show_baseline.data.mean(axis=-1)
                del epochs, freqs
                temp = freq_show.data.sum(axis=1)
                b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)

                freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])
                freq_show.data = 10* np.log10(freq_show.data/b_line[:, :, np.newaxis])
                freq_show.freqs = np.array([22.5])
            else:
                events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], stim)))
                
                try:
                    epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, picks = picks, preload = True)
                except:
                    print("Bad in", subj, pref, "stim")
                    raise Exception("I belive I can fly!!!")
                epochs.resample(300)
                freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = 8, use_fft = False, return_itc = False)
                freq_show_baseline = freq_show.copy().crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
                b_line  = freq_show_baseline.data.mean(axis=-1)
                           
                epochs.resample(300)
                del raw_data


                #b_line  = freq_show_baseline.data.mean(axis=-1)
                del epochs, freqs
                temp = freq_show.data.sum(axis=1)
                b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)

                freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])
                freq_show.data = 10 * np.log10(freq_show.data/b_line[:, :, np.newaxis])
                freq_show.freqs = np.array([22.5])
        elif "all" == stim: 
            events_w = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], "w", pref[1]))) 
            events_d = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], "d", pref[1]))) 
            events = np.concatenate((events_w,  events_d))
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, picks = picks, baseline=baseline, preload = True)
            except:
                print("Bad in", subj, pref)
                raise Exception("I belive I can fly!!!")
            epochs.resample(300)
            freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False) 
            
            freq_show_baseline = freq_show.copy().crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            b_line  = freq_show_baseline.data.mean(axis=-1)
            freq_show.data = 10 * np.log10(freq_show.data/b_line[:, :, np.newaxis])
            freq_show.freqs = np.array([22.5])
        elif stim == "r_wrong":
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}.txt".format(subj, pref[0], "wrong")))
            events_react = np.unique(read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}.txt".format(subj, pref[0], stim))), axis=0)

            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, picks = picks, preload = True)
            except:
                print("Bad in", subj, pref, "stim")
                raise Exception("I belive I can fly!!!")
            epochs.resample(300)

            freq_show_baseline = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False).crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            
            b_line  = freq_show_baseline.data.mean(axis=-1)
            del freq_show_baseline
            try:
                epochs = mne.Epochs(raw_data, events_react, event_id = None, tmin = period_start, tmax = period_end, picks = picks, preload = True)
            except:
                print("Bad in", subj, pref, "react")
                raise Exception("I belive I can fly!!!")            
            epochs.resample(300)
            del raw_data
            freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False)
            del epochs, freqs

            temp = freq_show.data.sum(axis=1)
            b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)

            freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])
            freq_show.data = np.log10(freq_show.data/b_line[:, np.newaxis])

        elif stim == "wrong":
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}.txt".format(subj, pref[0], stim)))

            #events = read_events(f"Cleaned_events_3/{subj}_{pref[0]}.txt")#mne.pick_events(events, include = [2, 6, 16, 4, 18, 20, 22, 32])
            
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, picks = picks, baseline=baseline, preload = True)
            except:
                print("Bad in", subj, pref)
                raise Exception("I belive I can fly!!!")
            del raw_data
            epochs.resample(300)
            
            #freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = 8, use_fft = False, return_itc = False)
            freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False)

            freq_show_baseline = freq_show.copy().crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            b_line  = freq_show_baseline.data.mean(axis=-1)
            
            temp = freq_show.data.sum(axis=1)
            freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])

            b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)
            freq_show.data =  np.log10(freq_show.data/b_line[:, np.newaxis])
            print(freq_show.data.shape)
        else:
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], "w", pref[1])))

            #events = read_events(f"Cleaned_events_3/{subj}_{pref[0]}.txt")#mne.pick_events(events, include = [2, 6, 16, 4, 18, 20, 22, 32])
            
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, picks = picks, baseline=baseline, preload = True)
            except:
                print("Bad in", subj, pref)
                raise Exception("I belive I can fly!!!")
            del raw_data
            epochs.resample(300)
            
            #freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = 8, use_fft = False, return_itc = False)
            freq_show = mne.time_frequency.tfr_multitaper(epochs, freqs = freqs, n_cycles = freqs//2, use_fft = False, return_itc = False)

            freq_show_baseline = freq_show.copy().crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            b_line  = freq_show_baseline.data.mean(axis=-1)
            
            temp = freq_show.data.sum(axis=1)
            freq_show.data = temp.reshape(temp.shape[0],1,temp.shape[1])

            b_line = b_line.sum(axis=1).reshape(temp.shape[0],1)
            freq_show.data = 10 * np.log10(freq_show.data/b_line[:, np.newaxis])
            freq_show.freqs = np.array([22.5])
        
        assert(freq_show.freqs.shape[0] == freq_show.data.shape[1])
        freq_show.save(os.path.join(os.getcwd(), "Sensor", "TFR", out_path, "{0}_{1}-{2}_{3}_int_50ms-tfr.h5".format(subj, pref[0], pref[1], stim)), overwrite=True)
        print(freq_show.freqs.shape, freq_show.data.shape)
        print(subj, pref, " ended!")

p_func = delayed(calculate_beta)
parallel = Parallel(6, max_nbytes = None)

answ = input(f"Are you sure? Your beta data in {out_path} will be rewrited.\n")
if  answ == "" or answ != "yes":
    print("Analisys canceled")
else:
    #calculate_beta(subjects[2] )
    parallel(p_func(subject) for subject in subjects)

