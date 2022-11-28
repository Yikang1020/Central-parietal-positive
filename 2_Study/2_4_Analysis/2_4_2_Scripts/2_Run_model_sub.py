import kabuki
import os
import hddm
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel,delayed
from glob import glob
from kabuki.analyze import gelman_rubin
import arviz
import numpy as np
from patsy import dmatrix 

from sklearn.preprocessing import StandardScaler



data = hddm.load_csv('./data.csv')
data['rt']=data['rt']+0.2
data['response']=data['_response']

df = data

df = hddm.utils.flip_errors(df)

for subj in df.subj_idx.unique():

    df.loc[df['subj_idx']==subj,'cpp_slope']=df.loc[df['subj_idx']==subj,'cpp_slope'].mean()

    df.loc[df['subj_idx']==subj,'cpp_amplitude']=df.loc[df['subj_idx']==subj,'cpp_amplitude'].mean()

    df.loc[df['subj_idx']==subj,'cpp_peak']=df.loc[df['subj_idx']==subj,'cpp_peak'].mean()

#df['cpp_slope']=StandardScaler().fit_transform(df['cpp_slope'].to_numpy().reshape(-1,1))

#df['cpp_amplitude']=StandardScaler().fit_transform(df['cpp_amplitude'].to_numpy().reshape(-1,1))

#df['cpp_peak']=StandardScaler().fit_transform(df['cpp_peak'].to_numpy().reshape(-1,1))



def z_link_func(x, data=df):
    
    return 1 / (1 + np.exp(-x.to_frame()))


fig = plt.figure()

ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')

for i, subj_data in df.groupby('subj_idx'):

    subj_data.rt.hist(bins=20, histtype='step', ax=ax)



## behaivor

# m0: Basic ddm without other parameters
def run_m0(id, df=None, samples=None, burn=None, thin=1,save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 
    
    mname  = save_name + '_chain_%i'%id    

    m = hddm.HDDM(data)
    
    #m.find_starting_values()
    
    m.sample(samples, burn=burn, thin=thin,dbname=dbname, db='pickle')
    
    m.save(mname)
    
    return m

m0 = Parallel(n_jobs = 8)(delayed(run_m0)(id = i, df = df, samples=10000,burn = 2000,thin=5, save_name = '2_4_2_2_temp/m0' ) for i in range(8))


# m1: Basic ddm with parameters [z,st,sv,sz]
def run_m1(id, df=None, samples=None, burn=None, thin = 1,save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 
    
    mname  = save_name + '_chain_%i'%id    

    m = hddm.HDDM(data,include=['z','st','sv','sz'])
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m1 = Parallel(n_jobs = 8)(delayed(run_m1)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m1' ) for i in range(8))

# m2: HDDMRegressor: v ~ C(coherency, Treatment('low')), with ['z','st','sv','sz']
def run_m2(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDMRegressor(data,"v ~ C(coherency, Treatment('low'))" ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m2 = Parallel(n_jobs = 8)(delayed(run_m2)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m2' ) for i in range(8))

# m3: HDDMRegressor: v ~ C(coherency, Treatment('low')), with ['z','st','sv','sz']
def run_m3(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDMRegressor(data,"v ~ C(coherency, Treatment('low'))" ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m3 = Parallel(n_jobs = 8)(delayed(run_m3)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m3' ) for i in range(8))







# m: HDDMRegressor: v~1+C(coherency, Treatment('low')), z~1+C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m4(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': lambda x:x}
    
    z_reg={'model': "z~1+C(prioritization, Treatment('no'))", 'link_func': z_link_func}

    m = hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m4 = Parallel(n_jobs = 8)(delayed(run_m4)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m4' ) for i in range(8))


# m: HDDMRegressor: v ~ 1 + C(coherency, Treatment('low')), z~1+C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m4(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': lambda x:x}
    
    z_reg={'model': "z~1+C(prioritization, Treatment('no'))", 'link_func': z_link_func}

    m = hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m4 = Parallel(n_jobs = 8)(delayed(run_m4)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m4' ) for i in range(8))




# m: HDDMRegressor: v ~ 1 + C(coherency, Treatment('low')), t ~ 1 + C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m5(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': lambda x:x}
    
    t_reg={'model': "z~1+C(prioritization, Treatment('no'))", 'link_func': lambda x:x}

    m = hddm.HDDMRegressor(data, [v_reg, t_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m5 = Parallel(n_jobs = 8)(delayed(run_m5)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m5' ) for i in range(8))


## joint modeling

# m: HDDMRegressor: v ~ 1 + C(coherency, Treatment('low')), t ~ 1 + C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m6(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~1+C(coherency, Treatment('low'))+cpp_slope+cpp_slope:C(coherency, Treatment('low'))", 'link_func': lambda x: x}
    
    t_reg={'model': "t~1+C(prioritization, Treatment('no'))", 'link_func':  lambda x: x}
    
    m=hddm.HDDMRegressor(df ,[v_reg, t_reg] ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)  
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m6 = Parallel(n_jobs = 8)(delayed(run_m6)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m6' ) for i in range(8))


# m: HDDMRegressor: v ~ 1 + C(coherency, Treatment('low')), t ~ 1 + C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m7(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~1+C(coherency, Treatment('low'))+cpp_amplitude+cpp_amplitude:C(coherency, Treatment('low'))", 'link_func': lambda x: x}
    
    t_reg={'model': "t~1+C(prioritization, Treatment('no'))", 'link_func':  lambda x: x}
    
    m=hddm.HDDMRegressor(df ,[v_reg, t_reg] ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)  
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m7 = Parallel(n_jobs = 8)(delayed(run_m7)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m7' ) for i in range(8))



# m: HDDMRegressor: v ~ 1 + C(coherency, Treatment('low')), t ~ 1 + C(prioritization, Treatment('no')), with ['z','st','sv','sz']
def run_m8(id, df=None, samples=None, burn=None,thin=1, save_name=None): 
    
    dbname = save_name + '_chain_%i.db'%id 

    mname  = save_name + '_chain_%i'%id    
   
    v_reg={'model': "v~1+C(coherency, Treatment('low'))+cpp_peak+cpp_peak:C(coherency, Treatment('low'))", 'link_func': lambda x: x}
    
    t_reg={'model': "t~1+C(prioritization, Treatment('no'))", 'link_func':  lambda x: x}
    
    m=hddm.HDDMRegressor(df ,[v_reg, t_reg] ,include=['z'],keep_regressor_trace=True, group_only_regressors = False)  
    
    m.find_starting_values()
    
    m.sample(samples, burn=burn, dbname=dbname, thin=thin,db='pickle') 
    
    m.save(mname)
    
    return m

m8 = Parallel(n_jobs = 8)(delayed(run_m8)(id = i, df = df, samples=10000,burn = 2000,thin=2, save_name = '2_4_2_2_temp/m8' ) for i in range(8))
