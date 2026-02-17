import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')

df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)   
df=df.drop(['day','month','year'],axis=1)     

print(df.head())
print(df.tail())
print(df.info())

# Dependent and independent features

x=df.drop('FWI',axis=1)   
y=df['FWI']               

print(x.head())
print(y.head())

# Train Test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)

# Feature selection  (Multicoliniarity is a part of feature selection)

print(x_train.corr()) # correlation check

# check Multicoliniratity

plt.figure(figsize=(12,10)) 
corr=x_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

print(x_train.corr())

# multilinearity check

def correlation(dataset,threshold): 
    col_corr=set()          
    corr_matrix=dataset.corr()     
    for i in range(len(corr_matrix.columns)):  
        for j in range(i):     
            if abs(corr_matrix.iloc[i,j]) > threshold:   
                colname=corr_matrix.columns[i]  
                col_corr.add(colname)    
    return col_corr   

corr_features=correlation(x_train,0.85)   
print(corr_features) 

# After Multilinearity check drop the features for more then threshold value

x_train.drop(corr_features,axis=1,inplace=True)   
x_test.drop(corr_features,axis=1,inplace=True)   
print(x_train.shape)                      
print(x_test.shape)

# This Previous process called Feature Selection

# Feature Scalling or Standarization

from sklearn.preprocessing import StandardScaler   # module import
scaler=StandardScaler()                          
x_train_scaled=scaler.fit_transform(x_train)   
x_test_scaled=scaler.transform(x_test)          

print(x_train_scaled)    

# Box plot to understand effect of standard scaler

plt.subplots(figsize=(10,5))       
plt.subplot(1,2,1)                 
sns.boxplot(data=x_train)     
plt.title('X_train data Before scalling ')      
plt.subplot(1,2,2)                   
sns.boxplot(data=x_train_scaled)      
plt.title('X_train data After scalling ')
plt.show()

 #   Linear regression Model  : Predicting


from sklearn.linear_model import LinearRegression     
from sklearn.metrics import mean_absolute_error        
from sklearn.metrics import r2_score                    
linreg=LinearRegression()                         
linreg.fit(x_train_scaled,y_train)                
y_pred=linreg.predict(x_test_scaled)               
mae=mean_absolute_error(y_test,y_pred)           
score=r2_score(y_test,y_pred)                     
print("Mean absolute error: ",mae)
print("re score: ",score)


 # Lasso Regression Model  : L1 based  : feature selection

from sklearn.linear_model import Lasso      
from sklearn.metrics import mean_absolute_error         
from sklearn.metrics import r2_score                   
lasso=Lasso()                        
lasso.fit(x_train_scaled,y_train)                 
y_pred=lasso.predict(x_test_scaled)              
mae=mean_absolute_error(y_test,y_pred)            
score=r2_score(y_test,y_pred)                     
print("Mean absolute error: ",mae)
print("re score: ",score)


 #  Ridge Regression Model   : L2 based   : overfitting managing

from sklearn.linear_model import Ridge     
from sklearn.metrics import mean_absolute_error        
from sklearn.metrics import r2_score                    
ridge=Ridge()                        
ridge.fit(x_train_scaled,y_train)                
y_pred=ridge.predict(x_test_scaled)               
mae=mean_absolute_error(y_test,y_pred)             
score=r2_score(y_test,y_pred)                      
print("Mean absolute error: ",mae)
print("re score: ",score)


 # Elasticnet Regression Model  : Ridge and Lasso both

from sklearn.linear_model import ElasticNet     
from sklearn.metrics import mean_absolute_error         
from sklearn.metrics import r2_score                    
elnt=ElasticNet()                        
elnt.fit(x_train_scaled,y_train)                
y_pred=elnt.predict(x_test_scaled)               
mae=mean_absolute_error(y_test,y_pred)            
score=r2_score(y_test,y_pred)                      
print("Mean absolute error: ",mae)
print("re score: ",score)


#pickle file created : for deployment perpose
 
import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))      
pickle.dump(ridge,open('ridge.pkl','wb'))


