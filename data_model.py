import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cleaned_data.csv')

#choose relavant columns 
df.columns

df_model = df[['average_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

#get dummy data
# to create new binary columns for each unique category:
df_dum = pd.get_dummies(df_model)

#train-validate test splits 
from sklearn.model_selection import train_test_split

#X: The features (independent variables) of the dataset
#: The target variable (dependent variable) of the dataset

X = df_dum.drop('average_salary', axis =1)
y = df_dum.average_salary.values

#.value converts to an array data structure, which is prefered over series in modeling 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear regression

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
#multiple linear regression model is off on avg by about -20.8k using cross validation

#lasso regression
lm_l = Lasso(.12)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)
plt.show()

#lowest error value is -19.31 at alpha = 0.12

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

#random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

#off by -15.53, most effecint till now


#tune models using GridsearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
parameters = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [None, 5, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
    }
# Create the GridSearchCV object
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)

# Fit the GridSearchCV to the data
gs.fit(X_train, y_train)

# Get the best score and best estimator
best_score = gs.best_score_
best_estimator = gs.best_estimator_

print("Best Score:", best_score)
print("Best Estimator:", best_estimator)


#test ensabmles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)
mean_absolute_error(y_test,tpred_rf) #best


#tuned rf is best at 11.419208053691275