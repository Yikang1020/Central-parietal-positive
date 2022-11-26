## load package

## read data

## clean data
#### anova analysis

## extract feature
#### load EEG data
#### preprocessing

#### cpp peak
#### cpp slope
#### cpp amplitude
#### where is no cpp

## save data





%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import sys
import mne
from mne.event import define_target_events
from mne.channels import make_1020_channel_selections
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
import sklearn
import os
from glob import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from statsmodels.stats.anova import AnovaRM


print("Python version is", sys.version)
print("MNE version is", mne.__version__)

## read data
#subj_idx
subjects = ['sub-001', 
                 'sub-003', 
                 'sub-004', #
                 'sub-005', 
                 'sub-006', 
                 'sub-007', #
                 'sub-008', 
                 'sub-009', #
                 'sub-010', 
                 'sub-011', #
                 'sub-012',
                 'sub-013', #
                 'sub-014', #
                 'sub-015', 
                 'sub-016',
                 'sub-017']
# task
tasks=[]
for side in ['outside','inside']:
    task='sourcedata-eeg_'+side+'-MRT'
    tasks.append(task)
# runs
runs=[]
for task in tasks:
    if task == tasks[0]:
        side = 'outside'
        for i in range(1,3):
            run = side+'MRT_run-0'+str(i)+'_beh.tsv'
            runs.append(run)
    else:
        side='inside'
        for i in range(1,6):
            run = side+'MRT_run-0'+str(i)+'_beh.tsv'
            runs.append(run)

# df_dirs: path + subj_idx + task + datatype + run
df_dirs=[]
temp_dir =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
BIDS_data="2_Data\\Ostwald2018\\BIDS_data"
datatype='beh'
for subj_idx in subjects :
    for task in tasks:
        if task==tasks[0]:
            for run_index in range(0,2):
                run=runs[run_index]
                run=subj_idx+'_task-pdm_acq-'+run
                df_dir=os.path.join(temp_dir,BIDS_data,subj_idx,task,datatype,run)
                df_dirs.append(df_dir)
        else:
            pass

dfs=pd.DataFrame()
for df_dir in df_dirs:
    if os.path.exists(df_dir):
        df=pd.read_csv(df_dir,sep='\t')
        subject=''.join(re.findall(r'BIDS_data\\(.+?)\\sourcedata',df_dir))
        side=''.join(re.findall(r'\\sourcedata-eeg_(.+?)-MRT\\beh\\',df_dir))
        run=''.join(re.findall(r'run-0(.+?)_beh',df_dir))
        df['subject']=subject
        df['side']=side
        df['run']=int(run)
        dfs=pd.concat([df,dfs])
    else:
        pass

## clean data
dfs['attention'] = dfs['prioritization_cue'].map({74:'left',75:'right',76:'double'},na_action=None)
dfs['coherency']=dfs['condition'].map({1:'high',2:'high',3:'low',4:'low'})
dfs['prioritization']=dfs['condition'].map({1:'yes',2:'no',3:'yes',4:'no'})

car_images=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
face_images=[19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]
dfs['category']=dfs['image_index'].isin(car_images).astype(int).map({1:'car', 0:'face'})

bedata = dfs.loc[:,['response_time','response_corr','subject','attention','coherency','prioritization','category','run']]
bedata.rename(columns={'response_time':'rt','response_corr':'response','subject':'subj_idx','category':'stimulus'},inplace = True)
bedata.loc[(bedata['response']==1)&(bedata['stimulus']=='face'),'_response'] = 0 #face
bedata.loc[(bedata['response']==1)&(bedata['stimulus']=='car'),'_response'] = 1 #car
bedata.loc[(bedata['response']==0)&(bedata['stimulus']=='face'),'_response'] = 0
bedata.loc[(bedata['response']==0)&(bedata['stimulus']=='car'),'_response'] = 1
bedata['id']=bedata.index
bedata = bedata.sort_values(by=['run','id'], axis=0, ascending=True) #加入代码检查错误


# cpp
bedata['cpp_peak'] = np.nan
bedata['cpp_slope'] = np.nan
bedata['cpp_amplitude'] = np.nan

# ANOVA
model_aovrm2way = AnovaRM(bedata,
                   'response',
                   'subj_idx',
                   within=['coherency','prioritization','stimulus'],
                   aggregate_func='mean')
res2way=model_aovrm2way.fit()
print(res2way)
model_aovrm2way = AnovaRM(bedata,
                   'rt',
                   'subj_idx',
                   within=['coherency','prioritization','stimulus'],
                   aggregate_func='mean')
res2way=model_aovrm2way.fit()
print(res2way)


#event
event_dict = {
  'Response/car': 5,
  'Response/face': 6,
  'Stimulus/hc/p/left': 10,
  'Stimulus/hc/p/right': 11,
  'Stimulus/hc/np/left': 20,
  'Stimulus/hc/np/right': 21,
  'Stimulus/lc/p/left': 30,
  'Stimulus/lc/p/right': 31,
  'Stimulus/lc/np/left': 40,
  'Stimulus/lc/np/right': 41,
  'Cue/Left': 74,
  'Cue/Right': 75,
  'Cue/double': 76
}

cue_dict =  {
  'Cue/Left': 74,
  'Cue/Right': 75,
  'Cue/double': 76
}

stimulus_dict = {
  'Stimulus/hc/p/left': 10,
  'Stimulus/hc/p/right': 11,
  'Stimulus/hc/np/left': 20,
  'Stimulus/hc/np/right': 21,
  'Stimulus/lc/p/left': 30,
  'Stimulus/lc/p/right': 31,
  'Stimulus/lc/np/left': 40,
  'Stimulus/lc/np/right': 41,
}

response_dict = {
  'Response/car': 5,
  'Response/face': 6,
}

## extract feature
for subject in subjects: # except 004
    
    # read data
    side = 'sourcedata-eeg_outside-MRT'
    
    measures = ['eeg','beh']
    
    etask = os.path.join(subject + "_task-pdm_acq-outsideMRT_eeg.vhdr" )
    
    preprocessed = os.path.join(subject + "_pred.fif")
    
    edata_path = os.path.join(temp_dir, BIDS_data, subject, side, measures[0], etask)
    
    edata = mne.io.read_raw_brainvision(edata_path)

    # preprocessing
    edata.set_channel_types({'EOG':'eog'})   
    
    edata.set_channel_types({'ECG':'ecg'})
    
    edata.resample(512, npad="auto")
    
    edata.filter(1, 30, fir_design='firwin', picks=['eeg'])
    
    edata.set_eeg_reference('average', projection=True).apply_proj()

    # find events
    events_from_annot, event_dict = mne.events_from_annotations(edata)

    # ica remove artifact
    ica = mne.preprocessing.ICA(n_components=50, random_state=97)
    
    ica.fit(edata) 
    
    ica.exclude = []                                   
    eog_indices, eog_scores = ica.find_bads_eog(edata) 
                                                                         
    ecg_indices, ecg_scores = ica.find_bads_ecg(edata, method='ctps')
                                                                   
    ica.exclude = eog_indices + ecg_indices 
    
    ica.apply(edata) 
    
    # stimulus-locked epochs
    epochs = mne.Epochs(edata, events_from_annot, event_id=stimulus_dict, tmin=-1, tmax=1,
                    baseline = (None,0), preload=True, picks=['eeg'])

    tname = os.path.join(os.path.dirname(os.getcwd()),'2_4_3_tmp_data',subject+".fif")

    epochs.save(tname,overwrite=True)

    # response-locked epochs
    epochs_res = mne.Epochs(edata, events_from_annot, event_id=response_dict, tmin=-1, tmax=1,
                     baseline = (None,0), preload=True, picks=['eeg'])

    tname_res = os.path.join(os.path.dirname(os.getcwd()),'2_4_3_tmp_data',subject+"_res.fif")

    epochs_res.save(tname_res,overwrite=True)

    if subject != 'sub-004':

        # epochs_baseline
        channel = ['CPz','CP1','CP2']

        time = [-0.2, 0]


        # cpp peak
        time = [-0.25, -0.1]

        epochs_CPP = epochs_res.copy().pick_channels(channel)

        epochs_CPP = epochs_CPP.crop(time[0],time[1])

        times = epochs_CPP.times

        epochs_CPP = epochs_CPP.get_data()

        epochs_CPP = np.mean(epochs_CPP, axis = 1)

        CPP_peak = np.amax(epochs_CPP, axis = 1)

        # cpp slope
        time = [-0.25, -0.1]

        epochs_CPP = epochs_res.copy().pick_channels(channel)

        epochs_CPP = epochs_CPP.crop(time[0],time[1])

        times = epochs_CPP.times

        epochs_CPP = epochs_CPP.get_data()

        epochs_CPP = np.mean(epochs_CPP, axis = 1)

        CPP_slopes = []
        
        for i in range(epochs_CPP.shape[0]):
        
            CPP_slope = np.polyfit(times,epochs_CPP[i,:],1)[0]
        
            CPP_slopes = np.append(CPP_slopes,CPP_slope)

        channel = ['CPz','CP1','CP2']

        # cpp amplitude
        time = [-0.1, -0]

        epochs_CPP = epochs_res.copy().pick_channels(channel)

        epochs_CPP = epochs_CPP.crop(time[0],time[1])

        times = epochs_CPP.times

        epochs_CPP = epochs_CPP.get_data()

        epochs_CPP = np.mean(epochs_CPP, axis = 1) 

        CPP_amplitudes = np.mean(epochs_CPP, axis = 1)

        # which trial has cpp
        consecutives = []

        for trial in range(epochs_CPP.shape[0]):
            
            ps = []
            
            for i in range(len(times)):
                
                t,p = ttest_1samp(epochs_CPP[trial,:],0)
                
                if p<=0.05:
                    p = 1
                
                else:
                    p = 0
                ps = np.append(ps,p)
            
            start = 0
            
            end = start + 15
            
            consecutive = False

            while (end <= len(times)) and (consecutive == False): 
            
                if np.sum(ps[start:end]) == 15:
                    consecutive = True
            
                else:
                    start = start + 1
                    end = start + 15
            
            consecutives = np.append(consecutives,consecutive)

        CPP_peak[np.where(consecutives == 0)] = -1

        bedata.loc[bedata['subj_idx']==subject,'index'] = np.arange(288)
        # cpp in bedata
        where = np.array(np.where(bedata.loc[bedata['subj_idx']==subject,'rt'].isnull()))[0]

        bedata.loc[((bedata['subj_idx']==subject)&(bedata.loc[:,'index'].isin(np.setdiff1d(np.arange(288), where)))),'cpp_peak'] = CPP_peak

        bedata.loc[((bedata['subj_idx']==subject)&(bedata.loc[:,'index'].isin(np.setdiff1d(np.arange(288), where)))),'cpp_slope'] = CPP_slopes

        bedata.loc[((bedata['subj_idx']==subject)&(bedata.loc[:,'index'].isin(np.setdiff1d(np.arange(288), where)))),'cpp_amplitude'] = CPP_amplitudes
    
    
    else:
        pass

bedata['subj_idx']=bedata['subj_idx'].str.replace(r'sub-0',r'0')

bedata = bedata.reset_index(drop=True)

bedata=bedata.drop(bedata[bedata['cpp_peak']==-1].index)

bedata = bedata.dropna(axis=0,how='any').reset_index(drop=True)

## save data
bedata.to_csv('data.csv',index=False)
