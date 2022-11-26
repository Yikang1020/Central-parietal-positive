import hddm

models_0 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m0_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_0.append(modelx)
        

print(gelman_rubin(models_0))

m0 = kabuki.utils.concat_models(models_0)

m0.plot_posteriors()


m0.plot_posterior_predictive()

print(m0.dic)



models_1 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m1_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_1.append(modelx)
        

print(gelman_rubin(models_1))

m1 = kabuki.utils.concat_models(models_1)

m1.plot_posteriors()

m1.plot_posterior_predictive()

print(m1.dic)

models_2 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m2_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_2.append(modelx)
        

print(gelman_rubin(models_2))

m2 = kabuki.utils.concat_models(models_2)

m2.plot_posteriors()

m2.plot_posterior_predictive()

print(m2.dic)

models_3 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m3_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_3.append(modelx)
        

print(gelman_rubin(models_3))

m3 = kabuki.utils.concat_models(models_3)

m3.plot_posteriors()

m3.plot_posterior_predictive()

print(m3.dic)

models_4 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m4_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_4.append(modelx)
        

print(gelman_rubin(models_4))

m4 = kabuki.utils.concat_models(models_4)

m4.plot_posteriors()

m4.plot_posterior_predictive()

print(m4.dic)

models_5 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m5_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_5.append(modelx)
        

print(gelman_rubin(models_5))

m5 = kabuki.utils.concat_models(models_5)

m5.plot_posteriors()

m5.plot_posterior_predictive()

print(m5.dic)

models_6 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m6_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_6.append(modelx)
        

print(gelman_rubin(models_6))

m6 = kabuki.utils.concat_models(models_6)

m6.plot_posteriors()

m6.plot_posterior_predictive()

print(m6.dic)



models_7 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m7_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_7.append(modelx)
        

print(gelman_rubin(models_7))

m7 = kabuki.utils.concat_models(models_7)

m7.plot_posteriors()

m7.plot_posterior_predictive()

print(m7.dic)




models_8 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m8_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_8.append(modelx)
        

print(gelman_rubin(models_8))

m8 = kabuki.utils.concat_models(models_8)

m8.plot_posteriors()

m8.plot_posterior_predictive()

print(m8.dic)





models_9 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m9_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_9.append(modelx)
        

print(gelman_rubin(models_9))

m9 = kabuki.utils.concat_models(models_9)

m9.plot_posteriors()

m9.plot_posterior_predictive()

print(m9.dic)



models_10 = []
for model_path in glob(os.path.join(os.getcwd(),'2_4_2_2_temp/m10_chain_*')):
    if ('db' not in model_path) and ('csv' not in model_path):
        modelx = hddm.load(model_path)
        models_10.append(modelx)
        

print(gelman_rubin(models_10))

m10 = kabuki.utils.concat_models(models_10)

m10.plot_posteriors()

m10.plot_posterior_predictive()

print(m10.dic)


