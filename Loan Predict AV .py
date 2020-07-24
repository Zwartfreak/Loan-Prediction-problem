#!/usr/bin/env python
# coding: utf-8

# #    Loan Prediction solution

# In this notebook, I have used different classifications algorithms to demonstrate that it's very important to understand which algorithm/model is best suitable and for which problem.
# - **LogisticRegression**
# - **Naive Bayes**
# - **KNeighbors**
# - **Random Forest**
# - **SVM**

# In[69]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# In[267]:


#Loading the dataset

train=pd.read_csv(r"B:/Projects/Loan prediction problem/train.csv") 


# ## Data Analysis and Cleaning

# Displaying first 5 rows to get an overview of data

# In[268]:


train.head()


# In[269]:


train.shape


# In[270]:


train.count()


# In[271]:


train.dtypes


# **Finding missing values**

# In[272]:


train.isnull().sum()


# In[273]:


train.isnull().sum().sum()


# **Drop unnecessary columns/data***

# In[274]:


train = train.dropna(how='any', subset=['LoanAmount', 'Married', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History'])


# In[275]:


train.shape


# In[276]:


train.isnull().sum().sum()


# **Dealing missing values**

# In[277]:


train['Gender'] = train['Gender'].fillna(value='Can\'t say')
train['Dependents'] = train['Dependents'].fillna(value=0)
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(value=train['Loan_Amount_Term'])


# In[278]:


train.isnull().sum().sum()


# In[279]:


train.isnull().sum()


# In[280]:


cleaned_data = train


# In[281]:


cleaned_data.columns


# ### Dataset has been cleaned

# In[282]:


train.columns


# ## Data Visualization

# In[283]:


train['Loan_Status'].value_counts().plot.bar()


# In[284]:


sns.catplot(x="Gender", y="LoanAmount", data=train, color='aqua');


# **Male takes more loan than female or others**

# In[285]:


sns.catplot(x="Married", y="LoanAmount", data=train, color='yellow');


# In[286]:


sns.catplot(x="Education", y="LoanAmount", data=train, color='green');


# In[287]:


sns.catplot(x="Self_Employed", y="LoanAmount", data=train, color='orange');


# In[288]:


plot = train.plot.scatter('ApplicantIncome','LoanAmount')


# In[289]:


sns.regplot('ApplicantIncome','LoanAmount', data=train, color='red')


# **It is clearly visible from above regplot that:**
# - People with low income tend to take loan more
# - Also, people with high income take hogh amound of loan for big investments

# In[290]:


plt.figure(1) 
plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()


# **It can be inferred from the above bar plots that:**
# 
# - 80% applicants in the dataset are male.<br></br>
# - Around 65% of the applicants in the dataset are married.<br></br>
# - Around 15% applicants in the dataset are self employed.<br></br>
# - Around 85% applicants have repaid their debts.

# In[291]:


sns.distplot(train['ApplicantIncome'])


# **It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed**

# In[292]:


matrix = train.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[293]:


train.dtypes


# In[294]:


train = train.drop('Loan_ID', axis=1)


# #### **Now we need to `encode` our values from categorical to numerical data so that it would be easier for data modelling.**
# - We will use `LabelEncoder` for that

# In[295]:


le = LabelEncoder()

print('Gender : ',train['Gender'].unique())
train['Gender'] = le.fit_transform(train['Gender'])
print('Gender : ',train['Gender'].unique())
print('')
print('Married : ',train['Married'].unique())
train['Married'] = le.fit_transform(train['Married'])
print('Married : ',train['Married'].unique())
print('')
print('Education : ',train['Education'].unique())
train['Education'] = le.fit_transform(train['Education'])
print('Education : ',train['Education'].unique())
print('')
print('Self_Employed : ',train['Self_Employed'].unique())
train['Self_Employed'] = le.fit_transform(train['Self_Employed'])
print('Self_Employed : ',train['Self_Employed'].unique())
print('')
print('Property_Area : ',train['Property_Area'].unique())
train['Property_Area'] = le.fit_transform(train['Property_Area'])
print('Property_Area : ',train['Property_Area'].unique())


# #### Converting all object types to integer types

# In[296]:


train['Gender'] = train['Gender'].astype('int')
train['Married'] = train['Married'].astype('int')
train['Education'] = train['Education'].astype('int')
train['Self_Employed'] = train['Self_Employed'].astype('int')
train['Property_Area'] = train['Property_Area'].astype('int')
train['Property_Area'] = train['Property_Area'].astype('int')


# In[297]:


train['Dependents'] = train['Dependents'].replace('3+',3)


# In[298]:


train['Dependents'] = train['Dependents'].astype('int')


# In[299]:


train.dtypes


# In[300]:


print('Loan_Status : ',train['Loan_Status'].unique())
train['Loan_Status'] = le.fit_transform(train['Loan_Status'])
print('Loan_Status : ',train['Loan_Status'].unique())


# In[301]:


train['Loan_Status'] = train['Loan_Status'].astype('int')


# In[302]:


train.head()


# In[303]:


train['ApplicantIncome'] = train['ApplicantIncome']/1000
train['CoapplicantIncome'] = train['CoapplicantIncome']/1000
train['LoanAmount'] = train['LoanAmount']/1000
train['Loan_Amount_Term'] = train['Loan_Amount_Term']/1000


# ### Train-Test-Split

# In[304]:


y = train['Loan_Status']
x = train.drop(['Loan_Status'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=4)


# ## Data Modelling

# #### Logistic Regression

# In[305]:


lr = LogisticRegression()


# In[306]:


lr.fit(x_train,y_train)


# **Applying R squared method to evaluate the model**

# In[307]:


lr.score(x_test, y_test)*100


# In[308]:


lr.score(x_train, y_train)*100


# In[309]:


y_pred = lr.predict(x_test)


# ## Naive Bayes

# In[310]:


nb = GaussianNB()
nb.fit(x_train, y_train)


# In[311]:


nb.score(x_train, y_train)


# In[312]:


nb.score(x_test, y_test)


# ## KNN

# In[313]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[314]:


knn.fit(x_train, y_train)


# In[315]:


knn.score(x_test, y_test)


# ## Random Forest

# In[316]:


# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(x_train, y_train)   


# In[317]:


y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)


# In[318]:


display(regressor.score(x_test,y_test))


# In[319]:


display(regressor.score(x_train,y_train))


# ## Support Vector Machines (SVM)

# In[320]:


svc = svm.SVC(kernel ='linear', C = 1).fit(x_train, y_train) 


# In[321]:


svc.fit(x_train,y_train)


# In[322]:


svc.score(x_test, y_test)


# In[323]:


svc.score(x_train, y_train)


# ## All models together

# In[327]:


models = {'LR':LogisticRegression(), 'NB':GaussianNB(), 'KNN':KNeighborsClassifier(n_neighbors=10), 'RF':RandomForestRegressor(n_estimators = 100),'SVM':svm.SVC()}
accuracy_list = {}
for key,value in models.items():
    model = value
    model.fit(x_train, y_train)
    accuracy_list.update({key:model.score(x_test, y_test)})


# In[325]:


accuracy_list


# In[326]:


print('Best model is : `{}` '.format(max(accuracy_list, key=accuracy_list.get)))


# ## Additional knowledge

# Based on the domain knowledge, we can come up with new features that might affect the target variable. We will create the following three new features:
# 
# - **Total Income** - As discussed during bivariate analysis we will combine the Applicant Income and Coapplicant Income. If the total income is high, chances of loan approval might also be high.
# - **EMI** - EMI is the monthly amount to be paid by the applicant to repay the loan. Idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate the EMI by taking the ratio of loan amount with respect to loan amount term.
# - **Balance Income** - This is the income left after the EMI has been paid. Idea behind creating this variable is that if this value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval.
# 

# In[255]:


train['Total_Income']=(train['ApplicantIncome']+train['CoapplicantIncome'] )/1000


# In[256]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 


# In[257]:


train['Balance Income']=(train['Total_Income']-(train['EMI']*1000))/1000
# Multiply with 1000 to make the units equal


# Let us now drop the variables which we used to create these new features. Reason for doing this is, the correlation between those old features and these new features will be very high and logistic regression assumes that the variables are not highly correlated. We also wants to remove the noise from the dataset, so removing correlated features will help in reducing the noise too.
# 
# 

# In[258]:


train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 


# In[259]:


train.head()


# In[260]:


y = train['Loan_Status']
x = train.drop(['Loan_Status'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=4)


# In[261]:


lr.fit(x_train,y_train)


# In[262]:


lr.score(x_test, y_test)


# In[ ]:




