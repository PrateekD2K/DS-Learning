# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:16:22 2024

@author: Prateek.Mishra
"""

import pandas as pd
loans_df=pd.read_csv("C:/Users/Prateek.Mishra/Downloads/cs-training.csv")

columns_list = loans_df.columns.values
variable_datatype=loans_df.dtypes

### Analysis DataFrame ###

"""
Observation1 :- 1) SeriousDlqin2yrs is showing load defaulter so its my target variable
2)RevolvingUtilizationOfUnsecuredLines is showing credit utilization
3)age is showing age of the person
4) NumberOfTime30 is showing 1 month late freq
5) DebtRatio is showing dept to income ratio
6) MonthlyIncome
7)NumberOfOpenCreditLinesAndLoans is showing no of loans already exits
8) NumberOfTimes90DaysLate is showing 3 months late freq
9) NumberRealEstateLoansOrLines is showing house loans
10) NumberOfTime60 is showing 2 months late freq
11) NumberOfDependents is showing no of departments
"""
missing_values=loans_df.isnull().sum()

"""
Observation2 :- Issues in data
1) MonthlyIncome with missing values
2) No of Departments with missising value
3)dept ratio is very high 
"""

### Check Catagorical and Continious variable in dataframe for find analyzing the relationships b/w variable###
### Dedect Outliers ###

#for discrete variables (Catagorical)
loans_df['SeriousDlqin2yrs'].value_counts()
loans_df['age'].value_counts(sort=False)
""" some of age is more than 100 years so its also as issue of outliers"""

loans_df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts(sort=False)
loans_df['NumberOfTimes90DaysLate'].value_counts(sort=False)
loans_df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts(sort=False)
""" 98, 96 present it which is large number as compare to other and also present here more no of time"""

loans_df['NumberOfOpenCreditLinesAndLoans'].value_counts(sort=False)
loans_df['NumberRealEstateLoansOrLines'].value_counts(sort=False)
""" present large no of homeloan"""

loans_df['NumberOfDependents'].value_counts(sort=False)

#For Continious variable
loans_df['RevolvingUtilizationOfUnsecuredLines'].describe()
""" I can see mean and median are far away and max values also 50K so outliers present in data"""

util_percentiles=loans_df['RevolvingUtilizationOfUnsecuredLines'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(util_percentiles,2)

loans_df['MonthlyIncome'].describe()
missing_no=loans_df['MonthlyIncome'].isnull().sum()/len(loans_df)
persentage_of_missing_monthlyincome_values=missing_no*100 # Arround 20%

''' for dedecting outliers in continious variable we can check with box ploat as well'''
loans_df.boxplot(column="MonthlyIncome")
loans_df.boxplot(column="DebtRatio")

loans_df['DebtRatio'].describe()
missing_no=loans_df['DebtRatio'].isnull().sum()/len(loans_df)
persentage_of_missing_debtratio_values=missing_no*100 # Arround 20%

'''### Overall Analysis ###
#Data Present with outliers, missing values and default values'''

###############################################################################
############################### Data Cleaning #################################

# fpr RevolvingUtilizationOfUnsecuredLines
util_percentiles=loans_df['RevolvingUtilizationOfUnsecuredLines'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(util_percentiles,2)
''' here i can saw 97% data is clean so we can use median value to fill the data'''


median_value=loans_df['RevolvingUtilizationOfUnsecuredLines'].median()

temp_vect=loans_df['RevolvingUtilizationOfUnsecuredLines']>1
temp_vect.value_counts()

loans_df['util_new']=loans_df['RevolvingUtilizationOfUnsecuredLines']
#replace outliers with median values in new variable
loans_df['util_new'][temp_vect]=median_value 

#check with new veriable
loans_df['util_new'].describe()

percentile=loans_df['util_new'].quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.80, 0.9,0.91,0.95,0.96,0.97,0.975,0.98,0.99,1])
round(percentile,2)

# For late Freq with imputational base on target
freq_table_with_delay=loans_df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts(sort=False)

#freq_table_with_delay[13:len(freq_table_with_delay)]
#freq_table_with_delay[13:len(freq_table_with_delay)].sum()/freq_table_with_delay.sum()

''' I used cross tab to identify the outlier with compare to good or bad customer whixch present in SeriousDlqin2yrs column who's basically my target variable'''
cross_tab_target=pd.crosstab(loans_df['NumberOfTime30-59DaysPastDueNotWorse'],loans_df['SeriousDlqin2yrs'])

cross_tab_target_percent=cross_tab_target.astype(float).div(cross_tab_target.sum(axis=1), axis=0)
round(cross_tab_target_percent,2)
''' 0 showes the number of good customer and 1 showes the number of bad customer'''
'''Observation3 :- 98 with large number out outliers and its bad customer % nearly to 6 day freq so clean data according to that'''

loans_df['30_days_dpd_new']=loans_df['NumberOfTime30-59DaysPastDueNotWorse']

'''fill values with 6'''
loans_df['30_days_dpd_new'][loans_df['30_days_dpd_new']>12]=6
loans_df['30_days_dpd_new']
loans_df['30_days_dpd_new'].value_counts(sort=False)

'''
Same process  we can apply with 60 days and 90 days delays variable
'''
### 60 days delay ###
freq_table_with_delay=loans_df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts(sort=False)

cross_tab_target=pd.crosstab(loans_df['NumberOfTime60-89DaysPastDueNotWorse'],loans_df['SeriousDlqin2yrs'])

cross_tab_target_percent=cross_tab_target.astype(float).div(cross_tab_target.sum(axis=1), axis=0)
round(cross_tab_target_percent,2)
'''Observation :- 98 with large number out outliers and its bad customer % nearly to 7 day freq so clean data according to that'''

loans_df['60_days_dpd_new']=loans_df['NumberOfTime60-89DaysPastDueNotWorse']

'''fill values with 7'''
loans_df['60_days_dpd_new'][loans_df['60_days_dpd_new']>12]=7
loans_df['60_days_dpd_new'].value_counts(sort=False)

### 90 Days Delay ###
freq_table_with_delay=loans_df['NumberOfTimes90DaysLate'].value_counts(sort=False)

cross_tab_target=pd.crosstab(loans_df['NumberOfTimes90DaysLate'],loans_df['SeriousDlqin2yrs'])

cross_tab_target_percent=cross_tab_target.astype(float).div(cross_tab_target.sum(axis=1), axis=0)
round(cross_tab_target_percent,2)
'''Observation :- 98 with large number out outliers and its bad customer % nearly to 2-3 day freq so clean data according to that'''

loans_df['90_days_dpd_new']=loans_df['NumberOfTimes90DaysLate']

'''fill values with 2'''
loans_df['90_days_dpd_new'][loans_df['60_days_dpd_new']>11]=2
loans_df['60_days_dpd_new'].value_counts(sort=False)

#Monthly income
loans_df['MonthlyIncome'].isnull().sum()
loans_df['MonthlyIncome'].isnull().sum()/len(loans_df) #almost 20% so i use indicator 
loans_df['MonthlyIncome_ind']=1
loans_df['MonthlyIncome_ind'][loans_df['MonthlyIncome'].isnull()]=0
'''values present with value 1 and not present with value 0'''

loans_df['MonthlyIncome_ind'].value_counts(sort=False)
loans_df['MonthlyIncome_new']=loans_df['MonthlyIncome']
loans_df['MonthlyIncome_new'][loans_df['MonthlyIncome'].isnull()]=loans_df['MonthlyIncome'].median()
round(loans_df['MonthlyIncome_new'].describe())

#No of Depandent
loans_df['NumberOfDependents'].isnull().sum()
loans_df['NumberOfDependents'].isnull().sum()/len(loans_df) #2.6% so i no need to set indicatior indicator 
loans_df['NumberOfDependents_new']=loans_df['NumberOfDependents']
loans_df['NumberOfDependents_new'][loans_df['NumberOfDependents'].isnull()]=loans_df['NumberOfDependents'].median()
round(loans_df['NumberOfDependents'].describe())

