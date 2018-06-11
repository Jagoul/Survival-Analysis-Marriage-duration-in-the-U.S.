"""
Created on Thu Feb  1 16:23:29 2018
@author: Raed Abdel Sater
Title : Survival analysis : Surival time of Marriges in the U.S.
"""
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import AalenAdditiveFitter
from lifelines import CoxPHFitter
import patsy, random
import matplotlib.pyplot as plt
from lifelines.plotting import plot_lifetimes
import seaborn as sns

# For Windows
#path = 'C:\\Users\\raed\\Dropbox\\INSE - 6320\\Final Project\\divorce.dat'

# for Linux Machine
path= '/home/raed/Dropbox/INSE - 6320/Final Project/divorce.dat'
with open(path,'r') as f:
     next(f) # skip first row
     data = pd.DataFrame(l.strip().split() for l in f)

# Performing a series of transformation on the original file in order to get a clean and tidy dataset
data.columns = ['ID','Husband_Education','Husband_Race','Couple_Race','Duration','Divorce', 'X','Y']
data['Husband_Education'] = data['Husband_Education'].map(str)+ " "+ data['Husband_Race']
data.drop('Husband_Race',1, inplace=True)
data.columns = ['ID','Husband_Education','Husband_Race','Couple_Race','Duration','Divorce', 'X']
data['Husband_Education'] = np.where(data['Husband_Race']=='years',  data['Husband_Education'].map(str) + ' ' +'years', data['Husband_Education'])
data['Husband_Education'] = np.where(data['Husband_Education']=='< 12 years',data['Husband_Education'].str.replace('< 12 years', 'Less than 12 years' ),data['Husband_Education'])
data['Husband_Race'] =np.where(data['Husband_Race']=='years', data['Couple_Race'],data['Husband_Race'])
data['Husband_Race'] =np.where(data['Husband_Race']=='Yes', 'Black', 'Other Ethnic Groups')
data['Couple_Race'] = np.where((data['Duration']== 'Yes')| (data['Duration']=='No'),data['Duration'],data['Couple_Race'])
data['Couple_Race'] =np.where(data['Couple_Race'] == 'Yes', 'Mixed-Race', 'Same-Race')
data['Duration'] = np.where((data['Divorce']=='Yes')|(data['Divorce']=='No'),data['Duration'],data['Divorce'])
data['Divorce'] = np.where((data['X']=='Yes')|(data['X']=='No'),data['X'],data['Divorce'])
data['Divorce'] = np.where(data['Divorce']=='No', 0,1)
data.drop('X',1,inplace=True)
data.drop(data.tail(1).index, inplace=True)

#load poverty dataset of Marriage dissoulution, education level and individual income in the U.S
poverty = pd.read_excel('est16us.xls', header=3)
poverty.drop(poverty.index[0], inplace=True)

#Export data as CSV and create a backup
data.to_csv(path_or_buf='/home/raed/Dropbox/INSE - 6320/Final Project/Divorce.csv',sep=',',index=False)
data = pd.read_csv('/home/raed/Dropbox/INSE - 6320/Final Project/Divorce.csv')
data['Duration'] = np.round(data['Duration'])

# Adding columns state , house income, children and poverty percentage
def assign_state (row):
   if ((row['Husband_Race'] =='Black') & (row['Husband_Education']=='Less than 12 years')) :
      return 'MS'
   if ((row['Husband_Race'] == 'Black')& (row['Husband_Education']=='12-15 years')) :
      return 'AL'
   if ((row['Husband_Race'=='Black'])& (row['Husband_Education']=='16+ years')):
      return 'MD'
   if ((row['Husband_Race'=='Other Ethnic Groups'])& (row['Husband_Education']=='Less than 12 years')):
       return 'AL'
   if((row['Husband_Race'=='Other Ethnic Groups'])& (row['Husband_Education']=='12-15 years')):
       return 'MD'
   if((row['Husband_Race'=='Other Ethnic Groups'])& (row['Husband_Education']=='16+ years')):
       return 'other'
   return 'NH'
data['State'] = data.apply (lambda row: assign_state (row),axis=1)

def assign_income (row):
    if(row['State']=='MD'):
        return '67,500$ - 75,000$'
    if(row['State']=='MS'):
        return '39,000$ - 40,593$'
    if(row['State']=='AL'):
        return '42,830$ - 44,765$'
    if(row['State']=='NH'):
        return '66,532$ - 70,303$'
data['Household_Income_Range'] = data.apply (lambda row: assign_income (row),axis=1)

data.rename(columns={'State':'Abbreviation'}, inplace=True)

def assign_state_name (row):
    if(row['Abbreviation']=='MD'):
        return 'Maryland'
    if(row['Abbreviation']=='MS'):
        return 'Mississippi'
    if(row['Abbreviation']=='AL'):
        return 'Alabama'
    if(row['Abbreviation']=='NH'):
        return 'New Hampshire'
data['State'] = data.apply (lambda row: assign_state_name (row),axis=1)

def assign_children(row):
    if((row['Husband_Race']=='Black')& (row['Household_Income_Range']=='39,000$ - 40,593$')):
        return 'Have Children'
    if((row['Husband_Race']=='Black')& (row['Household_Income_Range']=='42,830$ - 44,765$')):
        return 'Have Children'
    if((row['Husband_Race']=='Black')& (row['Household_Income_Range']=='66,532$ - 70,303$')):
        return 'Have Children'
    if((row['Husband_Race']=='Black')& (row['Household_Income_Range']=='67,500$ - 75,000$') & (row['Husband_Education']=='16+ years')):
        return 'No Children'
    if((row['Husband_Race']=='Other Ethnic Groups')& (row['Household_Income_Range']=='39,000$ - 40,593$')):
        return 'Have Children'
    if((row['Husband_Race']=='Other Ethnic Groups')& (row['Household_Income_Range']=='42,830$ - 44,765$')):
        return 'Have Children'
    if((row['Husband_Race']=='Other Ethnic Groups')& (row['Household_Income_Range']=='66,532$ - 70,303$') & (row['Husband_Education'=='16+ years'])):
        return 'No Children'
    if((row['Husband_Race']=='Other Ethnic Groups')& (row['Household_Income_Range']=='66,532$ - 70,303$') & (row['Husband_Education'=='12-15 years'])):
        return 'Have Children'
    if((row['Husband_Race']=='Other Ethnic Groups')& (row['Household_Income_Range']=='66,532$ - 70,303$') & (row['Husband_Education'=='Less than 12 years'])):
        return 'Have Children'
    return 'No Children'
data['Has_Children'] = data.apply(lambda row: assign_children (row), axis=1)    

def assign_poverty(row):
    if(row['Abbreviation']=='MD'):
        return 9.7
    if(row['Abbreviation']=='MS'):
        return 21
    if(row['Abbreviation']=='NH'):
        return 7.6
    if(row['Abbreviation']=='AL'):
        return 17.2
    return 0
data['Poverty_Percentage'] = data.apply(lambda row : assign_poverty(row),axis=1)

#Assign random dates for each marriage 
random.seed(2018)
data['Marriage_Date'] = np.random.randint(1970,2018, size=len(data))

#slicing 100 observations to represent censorship
sns.set()
current_time = 40
sliced_lifetimes = data['Duration']
actual_observations = np.random.choice(sliced_lifetimes,50) 
observed_lifetime = np.minimum(actual_observations, current_time)
observed_lifetimes = observed_lifetime.tolist()
observed = actual_observations <= current_time

#Calling plot_esimate to visualize events
plt.xlim(0, 50)
plt.vlines(current_time, 0, 50, lw=2, linestyles='--')
plt.xlabel("Marital time (in years)")
plt.title("Divorce representation After "+ str(current_time) +" Years")
sns.set()
plot_lifetimes(actual_observations, observed)
print("Observed divorces at %d:\n" %current_time, observed_lifetimes)

#Distribution of couples according to their marital duration
marital_duration = data.groupby('Duration')['Divorce'].count()
sns.set()
plt.plot(marital_duration, linewidth=3.0, color='crimson')
plt.xlabel("Years of Marriage")
plt.ylabel("Couples Getting Divorced")
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Marriage_distribution.pdf')
plt.show()

# calculate the correlation matrix
corr = data.corr()
print(corr)
sns.heatmap(pd.crosstab(data.Duration, data.Poverty_Percentage))
plt.show()

#Plotting Correlation between features to determine which features are the most important
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
plt.show()


"""
#Fitting the model using KaplanMeiler
"""
kmf = KaplanMeierFitter()
kmf.fit(durations=data['Duration'],event_observed=data['Divorce'])
#Plotting survival function
kmf.survival_function_.plot(title='Marriage Survival Time in the U.S', legend=False, linewidth=3.0)
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival.pdf')
plt.show()
kmf.plot(title='Survival Time Estimates of Mariages and its Confidence Intervals', legend= False, linewidth=3.0, show_censors=True)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival_ConfidenceInterval.pdf')
plt.show()

# Representing Left-censored events in marriage dataset
#ax=plt.subplot(111)
#ix=data['Husband_Race'] == 'Black'
#kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], left_censorship=True)
#print(kmf.cumulative_density_)
#ax = kmf.plot(title='Left Censored Events', ax=ax, legend=False)
#plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Left_Censored_Events.pdf')
#plt.show()

#Plotting survival analysis based on Couple race
ax =plt.subplot(111)
for r in data['Couple_Race'].unique():
    ix= data['Couple_Race'] == r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    ax = kmf.plot(title='Survival Time Estimate by Couple Race', legend = True, ax=ax, linewidth=3.0)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Couple_Race.pdf')
plt.show()

ax =plt.subplot(111)
for r in data['Husband_Education'].unique():
    ix = data['Husband_Education'] ==r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax =kmf.plot(title='Survival time Estimate by Education Level', legend=True,ax=ax, linewidth=2.5)
    
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Education_level.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Husband_Race'].unique():
    ix= data['Husband_Race'] == r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label= r)
    sns.set()
    ax=kmf.plot(title='Survival time Estimate by Husband Ethnicity', ax=ax, linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Ethnicity.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['State'].unique():
    ix= data['State'] == r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label= r)
    sns.set()
    ax=kmf.plot(title='Survival time Estimate by State', ax=ax, linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/State.pdf')
plt.show()

state_names = data['State'].unique()
for i,state_name in enumerate(state_names):
    ax = plt.subplot(2,2,i+1)
    ix = data['State'] == state_name
    kmf.fit( data['Duration'].loc[ix], data['Divorce'].loc[ix], label=state_name)
    sns.set()
    kmf.plot(ax=ax, legend=False)
    plt.title(state_name)
    plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Marriage_States.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Household_Income_Range'].unique():
    ix = data['Household_Income_Range'] == r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    ax=kmf.plot(title='Mariage Survival Estimate by Income', ax=ax, linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Household_Income.pdf')
plt.show()

income_rates = data['Household_Income_Range'].unique()
for i,income_rate in enumerate(income_rates):
    ax = plt.subplot(2,2,i+1)
    ix = data['Household_Income_Range'] == income_rate
    kmf.fit( data['Duration'].loc[ix], data['Divorce'].loc[ix], label=state_name)
    sns.set()
    kmf.plot(ax=ax, legend=False)
    plt.title(income_rate)
    plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/income_States.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Has_Children'].unique():
    ix=data['Has_Children']==r
    kmf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=kmf.plot(title='Mariage Survival Estimate Based on Children', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Children.pdf')
plt.show()

naf = NelsonAalenFitter()
naf.fit(data['Duration'], data['Divorce'])
sns.set()
naf.plot(title='Cumulative hazard over time', legend=False)
print(naf.cumulative_hazard_.head(32))
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_function.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Couple_Race'].unique():
    ix=data['Couple_Race']==r
    naf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=naf.plot(title='Cumulative Hazard by Couple Race ', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_CoupleRace.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Husband_Race'].unique():
    ix=data['Husband_Race']==r
    naf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=naf.plot(title='Cumulative Hazard by Husband Ethnicity ', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_HusbandRace.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Household_Income_Range'].unique():
    ix=data['Household_Income_Range']==r
    naf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=naf.plot(title='Cumulative Hazard by Income Range ', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_Income.pdf')
plt.show()

income_levels = data['Household_Income_Range'].unique()
for i,income_level in enumerate(income_levels):
    ax = plt.subplot(2,2,i+1)
    ix = data['Household_Income_Range'] == income_level
    naf.fit( data['Duration'].loc[ix], data['Divorce'].loc[ix], label=income_level)
    sns.set()
    naf.plot(ax=ax, legend=False)
    plt.title(income_level)
    plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_for_each_State.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['Husband_Education'].unique():
    ix=data['Husband_Education']==r
    naf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=naf.plot(title='Cumulative Hazard by Husband\'s Education level ', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_HusbandEducation.pdf')
plt.show()

Education_levels = data['Husband_Education'].unique()
for i,education_level in enumerate(Education_levels):
    ax = plt.subplot(2,2,i+1)
    ix = data['Husband_Education'] == education_level
    naf.fit( data['Duration'].loc[ix], data['Divorce'].loc[ix], label=education_level)
    sns.set()
    naf.plot(ax=ax, legend=False)
    plt.title(education_level)
    plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_for_each_EducationLevel.pdf')
plt.show()

ax=plt.subplot(111)
for r in data['State'].unique():
    ix=data['State']==r
    naf.fit(data['Duration'].loc[ix], data['Divorce'].loc[ix], label=r)
    sns.set()
    ax=naf.plot(title='Cumulative Hazard - State ', ax=ax,linewidth=2.5)
#Export the figure
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_State.pdf')
plt.show()

state_names = data['State'].unique()
for i,state_name in enumerate(state_names):
    ax = plt.subplot(2,2,i+1)
    ix = data['State'] == state_name
    naf.fit( data['Duration'].loc[ix], data['Divorce'].loc[ix], label=state_name)
    sns.set()
    naf.plot(ax=ax, legend=False)
    plt.title(state_name)
    plt.xlim(0, 80)
plt.tight_layout()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Cumulative_Hazard_for_each_State.pdf')
plt.show()

#Survival Regression using the following covariates : Couple Race, Income Range, State and Marriage Date
X = patsy.dmatrix('State + Couple_Race + Household_Income_Range + Husband_Education + Husband_Race + Marriage_Date -1', data, return_type='dataframe')
aaf = AalenAdditiveFitter(coef_penalizer=1.0, fit_intercept=True)
X['T'] = data['Duration']
X['E'] = data['Divorce']
aaf.fit(X, 'T', event_col='E')

aaf.cumulative_hazards_.head()
sns.set()
aaf.plot( columns=[ 'State[Alabama]', 'baseline','Couple_Race[T.Same-Race]','Household_Income_Range[T.42,830$ - 44,765$]'], ix=slice(1,15))
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival_Regression_for_Alabamae.pdf')
plt.show()

aaf.plot( columns=[ 'State[Mississippi]', 'baseline','Couple_Race[T.Same-Race]','Household_Income_Range[T.42,830$ - 44,765$]'], ix=slice(1,15))
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival_Regression_for_Mississippi.pdf')
plt.show()

aaf.plot( columns=[ 'State[New Hampshire]', 'baseline','Couple_Race[T.Same-Race]','Household_Income_Range[T.42,830$ - 44,765$]'],ix=slice(1,15))
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival_Regression_for_New_Hampshire.pdf')
plt.show()

aaf.plot( columns=[ 'State[Maryland]', 'baseline','Couple_Race[T.Same-Race]','Household_Income_Range[T.42,830$ - 44,765$]'],ix=slice(1,15))
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/Survival_Regression_for_Maryland.pdf')
plt.show()

#Regression is most interesting if we use it on data we have not yet seen, i.e. prediction! We can use what we have
#learned to predict individual hazard rates, survival functions, and median survival time. The dataset we are using is
#limited to 2017, so let’s use this data to predict the (though already partly seen) possible duration of couples who get married in 2017
#In the 4 different states which we are studying
#First we select 4 random couples married in 2017 each of which from a different state
# We are subsutting the dataframe 'data' to a smaller dataframe of 4 cpuples only and compare our prediction outputs'
row = {'State[Alabama]':[0.0] , 'State[Maryland]':[1.0],  'State[Mississippi]':[0.0], 'State[New Hampshire]':[0.0],  \
   'Couple_Race[T.Same-Race]':[1.0], 'Household_Income_Range[T.42,830$ - 44,765$]':[0.0],  \
   'Household_Income_Range[T.66,532$ - 70,303$]':[0.0],'Household_Income_Range[T.67,500$ - 75,000$]':[1.0], \
   'Husband_Education[T.16+ years]':[1.0] , 'Husband_Education[T.Less than 12 years]':[0.0], \
   'Husband_Race[T.Other Ethnic Groups]':[1.0],  'Marriage_Date':[2016], 'T':1,  'E':[0]}
MD = pd.DataFrame(data=row)
print("MD couple's unique data point", MD)

##plotting the predicted value for this specific couple
ax = plt.subplot(2,1,1)
aaf.predict_cumulative_hazard(MD).plot(ax=ax, legend=False)
plt.title('Mississippi Couple predicted Hazard and Survival time')
ax = plt.subplot(2,1,2)
aaf.predict_survival_function(MD).plot(ax=ax, legend=False);
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/MarylandCouple.pdf')
plt.show()

#same idea for Albama couple , we choose the same education level , ethnicity to keep our comparison valid
row = {'State[Alabama]':[1.0] , 'State[Maryland]':[0.0],  'State[Mississippi]':[0.0], 'State[New Hampshire]':[0.0],  \
   'Couple_Race[T.Same-Race]':[1.0], 'Household_Income_Range[T.42,830$ - 44,765$]':[1.0],  \
   'Household_Income_Range[T.66,532$ - 70,303$]':[0.0],'Household_Income_Range[T.67,500$ - 75,000$]':[0.0], \
   'Husband_Education[T.16+ years]':[0.0] , 'Husband_Education[T.Less than 12 years]':[0.0], \
   'Husband_Race[T.Other Ethnic Groups]':[1.0],  'Marriage_Date':[2016], 'T':1,  'E':[0]}
AL = pd.DataFrame(data=row)
print("AL couple's unique data point", AL)

##plotting the predicted value for this specific couple
ax = plt.subplot(2,1,1)
aaf.predict_cumulative_hazard(AL).plot(ax=ax, legend=False)
plt.title('Alabama Couple predicted Hazard and Survival time')
ax = plt.subplot(2,1,2)
aaf.predict_survival_function(AL).plot(ax=ax, legend=False);
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/AlabamaCouple.pdf')
plt.show()

#same idea for NewHampshire couple , we choose the same education level , ethnicity to keep our comparison valid
row = {'State[Alabama]':[0.0] , 'State[Maryland]':[0.0],  'State[Mississippi]':[0.0], 'State[New Hampshire]':[1.0],  \
   'Couple_Race[T.Same-Race]':[1.0], 'Household_Income_Range[T.42,830$ - 44,765$]':[1.0],  \
   'Household_Income_Range[T.66,532$ - 70,303$]':[0.0],'Household_Income_Range[T.67,500$ - 75,000$]':[1.0], \
   'Husband_Education[T.16+ years]':[1.0] , 'Husband_Education[T.Less than 12 years]':[1.0], \
   'Husband_Race[T.Other Ethnic Groups]':[1.0],  'Marriage_Date':[2015], 'T':2,  'E':[0]}
NH = pd.DataFrame(data=row)
print("AL couple's unique data point", NH)

##plotting the predicted value for this specific couple
ax = plt.subplot(2,1,1)
aaf.predict_cumulative_hazard(NH).plot(ax=ax, legend=False)
plt.title('NewHampshire Couple predicted Hazard and Survival time')
ax = plt.subplot(2,1,2)
aaf.predict_survival_function(NH).plot(ax=ax, legend=False);
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/NewHampshireCouple.pdf')
plt.show()

#same idea for Mississippi couple , we choose the same education level , ethnicity to keep our comparison valid
row = {'State[Alabama]':[0.0] , 'State[Maryland]':[0.0],  'State[Mississippi]':[1.0], 'State[New Hampshire]':[1.0],  \
   'Couple_Race[T.Same-Race]':[1.0], 'Household_Income_Range[T.42,830$ - 44,765$]':[0.0],  \
   'Household_Income_Range[T.66,532$ - 70,303$]':[0.0],'Household_Income_Range[T.67,500$ - 75,000$]':[0.0], \
   'Husband_Education[T.16+ years]':[0.0] , 'Husband_Education[T.Less than 12 years]':[1.0], \
   'Husband_Race[T.Other Ethnic Groups]':[1.0],  'Marriage_Date':[2016], 'T':1,  'E':[0]}
MS = pd.DataFrame(data=row)
print("MS couple's unique data point", NH)

##plotting the predicted value for this specific couple
ax = plt.subplot(2,1,1)
aaf.predict_cumulative_hazard(MS).plot(ax=ax, legend=False)
plt.title('Maryland Couple predicted Hazard and Survival time')
ax = plt.subplot(2,1,2)
aaf.predict_survival_function(MS).plot(ax=ax, legend=False);
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/MississippiCouple.pdf')
plt.show()

# Calculating the Cox propotional hazards regression model.
#The idea behind the model is that the log-hazard of an individual is a linear function of their static covariates
# and a population-level baseline hazard that changes over time. Mathematically:
#Converting the columns in the dataframe "data" into nummercial
 # Performing a series of transformation on the original file in order to get a clean and tidy dataset
cph_data = data
cph_data['Husband_Education'] = np.where(cph_data['Husband_Education']=='Less than 12 years',0,\
        (np.where(cph_data['Husband_Education']=='12-15 years',1,2)))
cph_data['Husband_Race'] =np.where(cph_data['Husband_Race']=='Other Ethnic Groups', 0,1)
cph_data['Couple_Race'] = np.where(cph_data['Couple_Race']== 'Same-Race',0,1)
cph_data.drop(['Abbreviation','State'], axis=1, inplace=True)
cph_data['Has_Children'] =np.where(cph_data['Has_Children']=='Have Children', 1, 0)
cph_data['Household_Income_Range'] = np.where(cph_data['Household_Income_Range']=='42,830$ - 44,765$',0,\
        (np.where(cph_data['Household_Income_Range']=='66,532$ - 70,303$',1,2)))
print('---------CPH fitting starts her---------------- ')
# Fitting the model and plotting the corresponding prediction 
cph = CoxPHFitter()
cph.fit(cph_data, duration_col='Duration', event_col='Divorce', show_progress=True)
cph.print_summary()
sns.set()
cph.plot()
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/CPH_Coefficients_plot.pdf')
plt.show()

#Calculating the correlation between covariates in order to understand the relationship in the data
# calculate the correlation matrix
cph_correlation = cph_data.corr()
print(corr)
#sns.heatmap(pd.crosstab(data.Duration, data.Poverty_Percentage))
sns.pairplot(cph_correlation)
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/CPH_Correlation.pdf')
plt.show()

sns.set(style="white")
# Generate a large random dataset
# Compute the correlation matrix
corr = cph_data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(9, 7))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/CPH_Triangle_Correlation.pdf')
plt.show()

#The following is the loglogs curves of two variables in our Divore dataset. The first
#is the Same_race type, which does have (close to) parallel lines, hence satisfies our assumption:
Educaton_less12 = cph_data.loc[cph_data['Husband_Education'] == 0]
Educaton_12 = cph_data.loc[cph_data['Husband_Education'] == 1]
Educaton_16plus = cph_data.loc[cph_data['Husband_Education'] == 2]

has_children =cph_data.loc[cph_data['Has_Children']==1]
does_not_have_children = cph_data.loc[cph_data['Has_Children']==0]

kmf0 = KaplanMeierFitter()
kmf0.fit(Educaton_less12['Duration'], event_observed=Educaton_less12['Divorce'])

kmf1 = KaplanMeierFitter()
kmf1.fit(Educaton_12['Duration'], event_observed=Educaton_12['Divorce'])

kmf2 = KaplanMeierFitter()
kmf2.fit(Educaton_16plus['Duration'], event_observed=Educaton_16plus['Divorce'])

fig, axes = plt.subplots()
kmf0.plot_loglogs(ax=axes)
kmf1.plot_loglogs(ax=axes)
kmf2.plot_loglogs(ax=axes)

axes.legend(['Less Than 12', '12 years', '16 years and above'])
plt.savefig('/home/raed/Dropbox/INSE - 6320/Final Project/CPH_Pproportional_Hazards_Assumption.pdf')
plt.show()

kmf3 = KaplanMeierFitter()
kmf3.fit(has_children['Duration'], event_observed=has_children['Divorce'])

kmf4 = KaplanMeierFitter()
kmf4.fit(does_not_have_children['Duration'], event_observed=does_not_have_children['Divorce'])

fig, axes = plt.subplots()
kmf3.plot_loglogs(ax=axes)
kmf4.plot_loglogs(ax=axes)
axes.legend(['Has Children', 'Does not Have Children'])

plt.show()

#A quick and visual way to check the proportional hazards assumption of a variable is to plot
#the survival curves segmented by the values of the variable. If the survival curves are the same “shape”,
#and differ only by constant factor, then the assumption holds. A more clear way to see this is to plot
#what’s called the loglogs curve: the log(-log(survival curve)) vs log(time). If the curves are parallel
#(and hence do not cross each other), then it’s likely the variable satisfies the assumption. If the curves do cross,
#likely you’ll have to “stratify” the variable (see next section). In lifelines, the KaplanMeierFitter object has a .
#plot_loglogs function for this purpose.
#The following is the loglogs curves of two variables in our regime dataset. 
#The first is the democracy type, which does have (close to) parallel lines, hence satisfies our assumption:
cph.fit(cph_data, duration_col='Duration', event_col='Divorce', strata=['Has_Children'])
cph.print_summary()  # access the results using cph.summary
cph.plot()