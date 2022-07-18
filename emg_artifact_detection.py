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
os.makedirs("Cleaned_events_4", exist_ok=True)
init()

def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b)


def no_mio_events(epochs_ar, thres = 5):
    #N_events, N_chans, N_times
    epochs_ar = epochs_ar.swapaxes(0, 1)
    epochs_ar = epochs_ar.swapaxes(1, 2)
    N_chans, N_times, N_events = epochs_ar.shape
    
    EV_DATA = np.abs(epochs_ar)
    
    CHAN_MAX = np.max(EV_DATA, axis=1)
    RESH_DATA = EV_DATA.reshape(N_chans,  N_events* N_times) 
    
    CHAN_MEAN = np.mean(RESH_DATA, axis=1)
    CHAN_STD = np.std(RESH_DATA, axis=1)
    
    good_chan = []
    
    for i in range(N_events):
        select = np.where(CHAN_MAX[:, i] > CHAN_MEAN + thres*CHAN_STD)
        if select[0].shape[0] <= N_chans//4:
            good_chan.append(i)
    return np.array(good_chan)
        
    

data_path = '/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/RAW_trans/'
out_path = f'beta_{stim + "NEW"}_{L_freq}_{H_freq}/'
os.makedirs(os.path.join(os.getcwd(), "Sensor", "TFR", out_path), exist_ok = True)

with open("configure.py", "r") as f_in:
    settings = f_in.readlines()
    with open(os.path.join(os.getcwd(), "Sensor", "TFR", out_path, "confige.log"), "w") as f_out:
        f_out.writelines(settings)

with open("check_channels.txt", "r") as f_ch:
    ch_names = list(map(str.strip, f_ch.readlines()))
    
def rms(array, axis):
    return np.sqrt(np.mean(np.square(array), axis=axis))

def  event_correction(event):
    if event > 8320:
        event -= 8320
    elif event > 4160 and event != 8320:
        event -= 4160
    return event

vec_event_correction = np.vectorize(event_correction)
array = []
ind = 0
pref = session
print(os.path.join(os.getcwd(),f"{pref[0]}_{pref[1]}.txt"))
with open(os.path.join(os.getcwd(),f"{pref[0]}_{pref[1]}.txt"), "w") as fb:
    fb.write("\n")


def calculate_beta(subj):
    
    mne.set_log_level("ERROR")
    pref = session
    freqs = np.arange(L_freq, H_freq, f_step)
        
    raw_file = os.path.join("/net/server/mnt/home/inside/filtered", "{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0]))
    try:
        raw_data = mne.io.Raw(raw_file, preload=False)
    except:
        print("\t", "File wasn't found. Please, check the filename format")
        return 0

    picks = mne.pick_types(raw_data.info, meg = True)
    events_raw = mne.find_events(raw_data, stim_channel='STI101', output='onset', consecutive='increasing', min_duration=0, shortest_event=1, mask=None, uint_cast=False, mask_type='and', initial_event=False, verbose=None)
    events_raw[:,2] = vec_event_correction(events_raw[:,2])
    events = mne.pick_events(events_raw, include = [2, 6, 16, 4, 18, 20, 22, 32])
    
    epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, picks=picks, preload = True)
    epochs = epochs.pick(picks="grad")
    '''
    epochs_data_std = np.std(epochs.get_data(), axis=-1)
    epochs_data_std_max = np.max(epochs_data_std)
    '''
    thres = 7
    clean = no_mio_events(epochs.get_data(), thres = thres)
    #epochs_data_max_mean = np.mean(epochs_data_max, axis=0)
    #epochs_data_max_mean_std = np.std(epochs_data_max_mean)
    

    #thres = epochs_data_std_max  * n
    #thres = epochs_data_max_mean_std * n
    #epochs_new = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, baseline = baseline, picks=picks, preload = True, reject=dict(grad=thres))
   

    print(subj, thres, str(events.shape[0]) + " ~~~~ " + str(events[clean].shape[0]))

    with open(f"{pref[0]}_{pref[1]}.txt", "a") as fb:
        fb.write(f"{events.shape[0]} {events[clean].shape[0]}\n")
    '''
    cleaned = events[clean]
    del epochs, events, clean, picks, raw_data 
    full_ev = []
    for i in range(cleaned.shape[0]):
        safity = 0
        event_ind = events_raw.tolist().index(cleaned[i].tolist())
        while events_raw[event_ind, 2] != 34 and events_raw[event_ind, 2] != 36:
            full_ev.append(events_raw[event_ind].tolist())
            event_ind += 1
            safity += 1
            if safity > 5:
                print(subj, "is broken")
                print(i, "from", cleaned.shape[0])
                print(event_ind, "from", events_raw.shape[0])
                raise Exception("WTF!!!!!")
        full_ev.append(events_raw[event_ind].tolist())
    
    del events_raw
    evee = list(map(str, full_ev))
    evee = list(map(lambda x: x + "\n", evee))
    '''
    #print(subj, f"max in signal {np.max(np.abs(raw_data.get_data()[:-1]))}", f"thres: {thres}", f"now {temp.shape[0]}", f"before {events.__len__()}", sep="\n") 
    '''
    with open(f"Cleaned_events_4/{subj}_{pref[0]}.txt", "w") as ff:
        ff.writelines(evee)
    '''
   
    #print(subj, pref, " ended!")

p_func = delayed(calculate_beta)
parallel = Parallel(6, max_nbytes = None)

answ = input(f"Are you sure? Your beta data in {out_path} will be rewrited.\n")
if  answ == "" or answ != "yes":
    print("Analisys canceled")
else:
    print(*session)
    #calculate_beta(subjects[2] )
    
    parallel(p_func(subject) for subject in subjects)



