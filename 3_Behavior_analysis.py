from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


data = pd.read_csv('data.csv')

# sum
y = data['rt']

x1 = data['cpp_amplitude']
x1 = sm.add_constant(x1)

x2 = data['cpp_peak']
x2 = sm.add_constant(x2)

x3 = data['cpp_slope']
x3 = sm.add_constant(x3)

model1 = sm.OLS(y,x1).fit()
print(model1.summary())

model2 = sm.OLS(y,x2).fit()
print(model2.summary())

model3 = sm.OLS(y,x3).fit()
print(model3.summary())

print(pearsonr(x1[:,1],y),
pearsonr(x2[:,1],y),
pearsonr(x3[:,1],y))

predict1 = model1.predict()
plt.scatter(x1['cpp_amplitude'],y,label='true')
plt.plot(x1['cpp_amplitude'],predict1,label='fake',color='red')
plt.legend()
plt.show()

predict2 = model2.predict()
plt.scatter(x2['cpp_peak'],y,label='true')
plt.plot(x2['cpp_peak'],predict2,label='fake',color='red')
plt.legend()
plt.show()

predict3 = model3.predict()
plt.scatter(x3['cpp_slope'],y,label='true')
plt.plot(x3['cpp_slope'],predict3,label='fake',color='red')
plt.legend()
plt.show()

# subject
y = data[['subj_idx','rt']].groupby('subj_idx').mean()

x1 = data[['subj_idx','cpp_amplitude']].groupby('subj_idx').mean()
x1 = sm.add_constant(x1)

x2 = data[['subj_idx','cpp_peak']].groupby('subj_idx').mean()
x2 = sm.add_constant(x2)

x3 = data[['subj_idx','cpp_slope']].groupby('subj_idx').mean()
x3 = sm.add_constant(x3)

model1 = sm.OLS(y,x1).fit()
print(model1.summary())

model2 = sm.OLS(y,x2).fit()
print(model2.summary())

model3 = sm.OLS(y,x3).fit()
print(model3.summary())

print(pearsonr(x1[:,1],y),
pearsonr(x2[:,1],y),
pearsonr(x3[:,1],y))

predict1 = model1.predict()
plt.scatter(x1['cpp_amplitude'],y,label='true')
plt.plot(x1['cpp_amplitude'],predict1,label='fake',color='red')
plt.legend()
plt.show()

predict2 = model2.predict()
plt.scatter(x2['cpp_peak'],y,label='true')
plt.plot(x2['cpp_peak'],predict2,label='fake',color='red')
plt.legend()
plt.show()

predict3 = model3.predict()
plt.scatter(x3['cpp_slope'],y,label='true')
plt.plot(x3['cpp_slope'],predict3,label='fake',color='red')
plt.legend()
plt.show()

# subject(normalization)

stdsc=StandardScaler()


y = data[['subj_idx','rt']].groupby('subj_idx').mean()
y = stdsc.fit_transform(y)

x1 = data[['subj_idx','cpp_amplitude']].groupby('subj_idx').mean()
x1 = stdsc.fit_transform(x1)
x1 = sm.add_constant(x1)

x2 = data[['subj_idx','cpp_peak']].groupby('subj_idx').mean()
x2 = stdsc.fit_transform(x2)
x2 = sm.add_constant(x2)

x3 = data[['subj_idx','cpp_slope']].groupby('subj_idx').mean()
x3 = stdsc.fit_transform(x3)
x3 = sm.add_constant(x3)

model1 = sm.OLS(y,x1).fit()
print(model1.summary())

model2 = sm.OLS(y,x2).fit()
print(model2.summary())

model3 = sm.OLS(y,x3).fit()
print(model3.summary())

print(pearsonr(x1[:,1],y),
pearsonr(x2[:,1],y),
pearsonr(x3[:,1],y))

predict1 = model1.predict()

plt.scatter(x1['cpp_amplitude'],y,label='true')
plt.plot(x1['cpp_amplitude'],predict1,label='fake',color='red')
plt.legend()
plt.show()

predict2 = model2.predict()
plt.scatter(x2['cpp_peak'],y,label='true')
plt.plot(x2['cpp_peak'],predict2,label='fake',color='red')
plt.legend()
plt.show()

predict3 = model3.predict()
plt.scatter(x3['cpp_slope'],y,label='true')
plt.plot(x3['cpp_slope'],predict3,label='fake',color='red')
plt.legend()
plt.show()


