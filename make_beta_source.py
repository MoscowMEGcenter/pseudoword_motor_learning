import mne, os, pickle
from mne.minimum_norm import read_inverse_operator, make_inverse_operator,source_band_induced_power
import numpy as np
from mne import set_config
from joblib import Parallel, delayed
from configure import *

set_config("MNE_MEMMAP_MIN_SIZE", "1M")
set_config("MNE_CACHE_DIR", ".tmp")

init()

def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "")
        b = b.split("\n")
        b = list(map(str.split, b))
        b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b)

data_path = '/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/RAW_trans/'
inv_path = 'inverse_data/'
out_path = f'beta_{stim}clean_{L_freq}_{H_freq}/'
os.makedirs(os.path.join(os.getcwd(), "Source", "SourceEstimate", out_path), exist_ok = True)

with open("configure.py", "r") as f_in:
    settings = f_in.readlines()
    with open(os.path.join(os.getcwd(), "Source", "SourceEstimate", out_path, "confige.log"), "w") as f_out:
        f_out.writelines(settings)


def calculate_beta(subj):
    print(subj)
    for pref in session:
        print("s")
        mne.set_log_level("ERROR")
        raw_file = os.path.join(data_path, subj,"{0}_{1}_raw_tsss_bads_trans.fif".format(subj,pref[0]))
        try:
            raw_data = mne.io.Raw(raw_file, preload=True)
        except:
            print("\t", "File wasn't found. Please, check the filename format")
            return 0
        picks = mne.pick_types(raw_data.info, meg = True, eog = True)
        bands = dict(beta=[L_freq, H_freq])
        if stim[0] == "react"[0]:
            
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], "w" + stim[1:])))
            events_react = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], stim)))
            
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 0.0, picks = picks, preload = True)
            except e:
                print("Bad in", subj, pref, "stim")
                print(e)
                raise Exception("I belive I can fly!!!")
            epochs.resample(100)
            inverse_operator = mne.minimum_norm.read_inverse_operator(os.path.join(os.getcwd(), "Source", "MISC", inv_path, "{0}_{1}_{2}_raw_tsss_bads_trans-ico-4-inv.fif".format(subj,pref[0], "word")))
            stc = mne.minimum_norm.source_band_induced_power(epochs.pick('meg'), inverse_operator, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"].crop(tmin=-0.5, tmax=-0.1, include_tmax=True)
            
            b_line  = stc.data.mean(axis=-1)
            del stc
            print(subj, "baseline_ended")
            try:
                epochs = mne.Epochs(raw_data, events_react, event_id = None, tmin = period_start, tmax = period_end, picks = picks, preload = True)
            except:
                print("Bad in", subj, pref, "react")
                raise Exception("I belive I can fly!!!")            
            epochs.resample(100)
            del raw_data
            stc = mne.minimum_norm.source_band_induced_power(epochs.pick('meg'), inverse_operator, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"]
            
            stc.data = np.log10(stc.data/b_line[:, np.newaxis])
        else:
            
            #events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], pref[1], "w" + stim[1:])))
            events = read_events(os.path.join(os.getcwd(), "EVENTS", "{0}_{1}_{2}_{3}.txt".format(subj, pref[0], stim[0], pref[1])))
            try:
                epochs = mne.Epochs(raw_data, events, event_id = None, tmin = period_start, tmax = period_end, baseline = baseline, picks = picks, preload = True)
            except Exception as e:
                print(e)
                print("Bad in", subj, pref, "stim")
                raise Exception("I belive I can fly!!!")
            #epochs.resample(300)
            inverse_operator = mne.minimum_norm.read_inverse_operator(os.path.join(os.getcwd(), "Source", "MISC", inv_path, "{0}_{1}_{2}_raw_tsss_bads_trans-ico-4-inv.fif".format(subj,pref[0], "word")))
            stc = mne.minimum_norm.source_band_induced_power(epochs.pick('meg'), inverse_operator, bands, use_fft=False, df = f_step, n_cycles = 8, baseline=baseline, baseline_mode="logratio")["beta"]
        
        #morph = mne.compute_source_morph(stc, subjects_dir=os.environ['SUBJECTS_DIR'], subject_to = "avg_platon_27sub")
        morph = mne.compute_source_morph(stc, subjects_dir="/net/synology/volume1/data/programs/ANYA/SPEECH_LEARN/FREESURF", subject_to = "avg_platon_27sub")
        stc = morph.apply(stc)

        stc.save(os.path.join(os.getcwd(),"Source", "SourceEstimate",  out_path, "{0}_{1}-{2}_{3}_int_50ms".format(subj,pref[0],pref[1], stim)))
        print(subj, pref, " ended!")

p_func = delayed(calculate_beta)
parallel = Parallel(5, max_nbytes = None)

answ = input(f"Are you sure? Your beta data in {out_path} will be rewrited.\n")
if  answ == "" or answ != "yes":
    print("Analisys canceled")
else:
    #calculate_beta(subjects[2])
    parallel(p_func(subject) for subject in subjects)





