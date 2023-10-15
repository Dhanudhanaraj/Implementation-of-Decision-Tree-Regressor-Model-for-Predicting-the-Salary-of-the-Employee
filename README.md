# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:

To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets

2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters

3.Train your model -Fit model to training data -Calculate mean salary value for each subset

4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters -Experiment with different hyperparameters to improve performance

6.Deploy your model Use model to make predictions on new data in real-world application.
## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Dhanumalya.D 
Register Number:212222230030  

```
```
import pandas as pd
df=pd.read_csv("Salary.csv")

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()

x=df[['Position','Level']]
y=df['Salary']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```
## Output:

### Initial dataset:
![Screenshot from 2023-10-15 10-24-47](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/a57f798f-67f8-4cd2-911c-412f2e481649)

### Data Info:
![Screenshot from 2023-10-15 10-24-56](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/491dd6c7-12ac-46f4-9a8f-9957b7957fe4)

### Optimization of null values:
![Screenshot from 2023-10-15 10-25-06](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/f11f5928-de20-4e47-aa49-0060db9a7c6b)

### Converting string literals to numericl values using label encoder:
![Screenshot from 2023-10-15 10-25-15](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/ad87db50-4b1f-4348-9f42-37c14e0a1c82)

### Mean Squared Error:
![Screenshot from 2023-10-15 10-25-31](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/748c950b-3683-4159-a5f9-5449a6bd0940)

### R2 (variance):
![Screenshot from 2023-10-15 10-25-39](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/1e1f1947-85e8-425f-a19f-ab3fc65b7998)

### Prediction:
![Screenshot from 2023-10-15 10-25-50](https://github.com/Dhanudhanaraj/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119218812/01663e0a-41c6-48cd-84ac-70e0708cff99)


## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
