'''
m0: (null model)
hddm(data)

m1: (basic model )
hddm(data, include=['z','st','sv','sz'])

Q1: Is drift rate vary with coherency
m2: (regressor model, v vary with coherency)
hddm.HDDMRegressor(data,"v ~ C(coherency, Treatment("low"))" ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m3: (regressor model, v vary with coherency +stimcoding )
v_reg={'model': 'v~1+C(coherency, Treatment("low"))', link_func: v_link_func}

hddm.HDDMRegressor(data, v_reg ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

Q2: what role of attention(assume m3 is best model, m4,m5,m6,m7 is based on the best model)
m4: (regressor model, v vary with coherency +stimcoding，z vary with coherency )
v_reg={'model': 'v~1+C(coherency, Treatment("low"))', link_func: v_link_func}

z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', link_func: lambda x:x}

hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m5: (regressor model, v vary with coherency，z vary with coherency +stimcoding)
v_reg={'model': 'v~1+C(coherency, Treatment("low"))', link_func: lambda x:x}

z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', link_func: z_link_func}

hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m6: (regressor model, v vary with coherency +stimcoding，t vary with coherency)
v_reg={'model': 'v~1+C(coherency, Treatment("low"))', link_func: v_link_func}

t_reg={'model': 't~1+C(prioritization, Treatment("no"))'}

hddm.HDDMRegressor(data, [v_reg, t_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m7: (regressor model, v vary with coherency，prioritization +stimcoding)
v_reg={'model': 'v~1+C(coherency, Treatment("low")+C(prioritization, Treatment("no") +C(coherency, Treatment("low"):C(prioritization, Treatment("no"))', link_func: v_link_func}

hddm.HDDMRegressor(data, v_reg ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

Q3: is cpp consist with evidence accumulation(assume m4 is best model, m8,m9,m10 is based on the best model)
m8: (regressor model, v vary with coherency, cpp_slope +stimcoding，z vary with coherency +stimcoding)
v_reg={'model': 'v~1+C(coherency, Treatment("low")+cpp_slope+cpp_slope:C(coherency, Treatment("low"))', link_func: v_link_func}

z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', link_func: lambda x:x}

hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m9: (regressor model, v vary with coherency, cpp_peak +stimcoding，z vary with coherency +stimcoding)
v_reg={'model': 'v~1+C(coherency, Treatment("low")+cpp_peak+cpp_peak:C(coherency, Treatment("low"))', link_func: v_link_func}

z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', link_func: lambda x:x}

hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

m10: (regressor model, v vary with coherency, cpp_amplitude+stimcoding，z vary with coherency +stimcoding)
v_reg={'model': 'v~1+C(coherency, Treatment("low")+cpp_amplitude+cpp_amplitude:C(coherency, Treatment("low"))', link_func: v_link_func}

z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', link_func: lambda x:x}

hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
'''

import kabuki
import os
import hddm
import matplotlib.pyplot as plt
import pickle
from functools import partial
from p_tqdm import p_map
from glob import glob
from kabuki.analyze import gelman_rubin
import arviz
import numpy as np
from patsy import dmatrix 

data = hddm.load_csv('/home/mw/input/cpp7572/data.csv')

df = data
samples = 16000
burn = 8000

def v_link_func(x, data=df):
    stim = (np.asarray(dmatrix('0 + C(s,[[1],[-1]])', {'s':data.stimulus.loc[x.index]},return_type = 'dataframe')))
    return x.to_frame() * stim

def z_link_func(x, data=df):
    stim = (dmatrix('0 + C(s, [[1], [-1]])', {'s': data.stimulus.loc[x.index]},return_type='dataframe'))
    return 1 / (1 + np.exp(-np.multiply(x.to_frame(), stim)))

def run_m0(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    

    m = hddm.HDDM(data)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle')
    m.save(mname)
    
    return m

m0 = p_map(partial(run_m0, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m0'),
                 range(8))

def run_m1(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    

    m = hddm.HDDM(data,include=['z','st','sv','sz'])
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') 
    m.save(mname)
    
    return m

m1 = p_map(partial(run_m1, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m1'),
                 range(8))

def run_m2(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
   
    m = hddm.HDDMRegressor(data,"v ~ C(coherency, Treatment('low'))" ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') 
    m.save(mname)
    
    return m
    
m2 = p_map(partial(run_m2, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m2'),
                 range(8))


def run_m3(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    v_reg = {'model':"v~1+C(coherency, Treatment('low'))",'link_func':v_link_func}
    m = hddm.HDDMRegressor(data,v_reg ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m3 = p_map(partial(run_m3, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m3'),
                 range(8))

def run_m4(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': lambda x:x}
    z_reg={'model': "z~1+C(prioritization, Treatment('no'))", 'link_func': z_link_func}
    m = hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m4 = p_map(partial(run_m4, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m4'),
                 range(8))


def run_m5(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': lambda x:x}
    z_reg={'model': "t~1+C(prioritization, Treatment('no'))",'link_func': z_link_func}
    m = hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m5 = p_map(partial(run_m5, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m5'),
                 range(8))


def run_m6(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))", 'link_func': v_link_func}
    t_reg={'model': "t~1+C(prioritization, Treatment('no'))", 'link_func': lambda x:x}
    m = hddm.HDDMRegressor(data, [v_reg, t_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m6 = p_map(partial(run_m6, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m6'),
                 range(8))

def run_m7(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': 'v~ 1 + C(coherency, Treatment("low")) + C(prioritization, Treatment("no")) + C(coherency, Treatment("low")): C(prioritization, Treatment("no"))', 'link_func': v_link_func}
    m=hddm.HDDMRegressor(data, v_reg ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)
    
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m7 = p_map(partial(run_m7, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m7'),
                 range(8))


def run_m8(id, df=None, samples=None, burn=None, save_name=None): 

    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    


    
    v_reg={'model': "v~1+C(coherency, Treatment('low'))+cpp_slope+cpp_slope:C(coherency, Treatment('low'))", 'link_func': v_link_func}
    z_reg={'model': "z~1+C(prioritization, Treatment('no'))", 'link_func': lambda x: x}
    m=hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)   

    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m

m8 = p_map(partial(run_m8, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m8'),
                 range(8))


def run_m9(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': 'v~1+C(coherency, Treatment("low"))+cpp_peak+cpp_peak:C(coherency, Treatment("low"))', 'link_func': v_link_func}
    z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', 'link_func': lambda x: x}
    m=hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False) 

    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') 
    m.save(mname)
    
    return m
m9 = p_map(partial(run_m9, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m9'),
                 range(8))


def run_m10(id, df=None, samples=None, burn=None, save_name=None): 
#     print('running model %i'%id);
    import hddm
    
    dbname = save_name + '_chain_%i.db'%id 
    mname  = save_name + '_chain_%i'%id    
    

    
    v_reg={'model': 'v~1+C(coherency, Treatment("low"))+cpp_amplitude+cpp_amplitude:C(coherency, Treatment("low"))', 'link_func': v_link_func}
    z_reg={'model': 'z~1+C(prioritization, Treatment("no"))', 'link_func': lambda x:x}
    m=hddm.HDDMRegressor(data, [v_reg, z_reg] ,include=['z','st','sv','sz'],keep_regressor_trace=True, group_only_regressors = False)

    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db='pickle') # it's neccessary to save the model data
    m.save(mname)
    
    return m
m10 = p_map(partial(run_m10, 
                         df = df, 
                         samples = samples,
                         burn = burn,
                         save_name = '/home/mw/project/m10'),
                 range(8))
