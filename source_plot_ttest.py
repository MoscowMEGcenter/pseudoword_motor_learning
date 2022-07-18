import os
import mne
import numpy as np
from scipy import stats
from configure import *
from statsmodels.stats import multitest as mul

def signed_p_val(t, p_val):

   if t >= 0:
      return 1 - p_val
   else:
      return -(1 - p_val)

def folder_gen(stimul, sessions):
    temp = ""
    for ses in sessions:
        temp += stimul[0] + ses[6:] + "_vs_"
    return temp[:-4]

vect_signed_p_val = np.vectorize(signed_p_val)



session = ["active1-st", "active2-end"]
stimul = ["word",  "react"]
#stimul = ["r_right_hand", "r_left_hand"]
ttest_result_file = '{0}_{1}_sub22_integ50'
subjects = [ 
    '030_koal',
    '051_vlro',
    '128_godz',
    '136_spar',
    '176_nama',
    '202_skol',
    '211_gnlu',
    '277_trev',
    '307_firo',
    '308_lodm',
    '317_arel',
    '355_slya',
    '372_skju',
    '389_revi',
    '390_shko',
    '394_tiev',
    '402_maev',
    '406_bial',
    '409_kodm',
    '415_yael',
    '436_buni',
    '383_laan']
#for i in range(2):
#for ses in session:
for stim in stimul:
    data_dir = os.path.join(os.getcwd(), "Source","SourceEstimate", f'beta_{stim}clean_{L_freq}_{H_freq}/')
    #data_dir = #[os.path.join(os.getcwd(), "Source","SourceEstimate", f'beta_{stimul[0]}clean_{L_freq}_{H_freq}/'),
	 #   os.path.join(os.getcwd(), "Source","SourceEstimate", f'beta_{stimul[1]}clean_{L_freq}_{H_freq}/')
#  ]
    stc_test = mne.read_source_estimate(os.path.join(data_dir, f"355_slya_active1-st_{stim}_int_50ms-rh.stc"))
    #stc_test = mne.read_source_estimate(os.path.join(data_dir[0], f"355_slya_active1-st_{stimul[0]}_int_50ms-rh.stc"))
    stc_test.resample(20)

    comp1_per_sub = np.zeros(shape=(len(subjects), stc_test.data.shape[0], stc_test.data.shape[1]))
    comp2_per_sub = np.zeros(shape=(len(subjects), stc_test.data.shape[0], stc_test.data.shape[1]))
    os.makedirs(os.path.join(data_dir, "p_val_fdr") , exist_ok = True)
    for ind, subj in enumerate(subjects):
        print(ind + 1, subj)
        temp1 = mne.read_source_estimate(os.path.join(data_dir, "{0}_{1}_{2}_int_50ms-lh.stc".format(subj, session[0], stim)))
        temp1.resample(20)
        comp1_per_sub[ind, :, :] = temp1.data
        temp2 = mne.read_source_estimate(os.path.join(data_dir, "{0}_{1}_{2}_int_50ms-lh.stc".format(subj, session[1], stim)))
        temp2.resample(20)
        comp2_per_sub[ind, :, :] = temp2.data
        

    print("calculation ttest")
    folder = folder_gen(stim, session) 
    #folder = f"{ses}_stimul[1]_vs_stimul[0]"#folder_gen(stim, session) 
    t_stat, p_val = stats.ttest_rel(comp2_per_sub, comp1_per_sub, axis=0)
    print(p_val.min(), p_val.mean(), p_val.max())
    print(t_stat.min(), t_stat.mean(), t_stat.max())
    width, height = p_val.shape
    p_val_resh = p_val.reshape(width * height)
    _, p_val = mul.fdrcorrection(p_val_resh)
    p_val = p_val.reshape((width, height))
    p_val = vect_signed_p_val(t_stat, p_val)
    p_val_stc = mne.SourceEstimate(data = p_val, vertices = stc_test.vertices ,  tmin = stc_test.tmin, tstep = stc_test.tstep)
    print(p_val_stc.data.min(), p_val_stc.data.mean(), p_val_stc.data.max())
    print("Brain's rendering")
    #os.environ["SUBJECTS_DIR"] = 'D:\\beta_data\\stc_beta\\freesurfer\\'
    new_dir = os.path.join(data_dir, "p_val_fdr", ttest_result_file.format("p-val",folder))
    p_val_stc.subject = 'avg_platon_27sub'
    p_val_stc.save(new_dir)

