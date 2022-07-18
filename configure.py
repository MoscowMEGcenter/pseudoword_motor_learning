import os

def init():
    os.makedirs(os.path.join(os.getcwd(),"Sensor","TFR"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),"Sensor","Evoked"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),"Source","SourceEstimate"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),"Source","MISC"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(),"Accel"), exist_ok=True)
    print("Start analysis")

stim = "react"
#stim = "w_right_hand"
#stim = "w_left_hand"
#stim = "react"
#stim = "wrong"
#stim =  "r_right_hand"
#stim = "r_left_hand" 

L_freq = 15
H_freq = 26

f_step = 2

period_start = -2.000
period_end  = 2.000

baseline = (-0.5, -0.1)
session = [ ('active1', "st"), ('active2', "end")]
#session = [ ('active2', "end")]   

#otr_s    = [0, 1, 2, 3,    5,    8,        11, 12,     14, 15,     17, 19, 20]
#nedost_s = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]

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
    '383_laan'
]
#subjects = [subjects[i] for i in range(len(subjects)) if i in otr_s]


