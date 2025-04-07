<H3>NAME: Meenu.S</H3>
<H3>REGISTER NO.: 212223230124</H3>
<H3>DATE: </H3>
<H1>EX. NO.1</H1>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
~~~
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('Churn_Modelling.csv')
print(df)x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
print(df.isnull().sum())
numeric_features=df.select_dtypes(include=['number'])
df.fillna(numeric_features.mean(),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df['Balance'].describe())
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(numeric_features))
print(df1)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
~~~


## OUTPUT:
![image](https://github.com/user-attachments/assets/f947c35d-47d6-49a9-ad7e-79d794799b5d)
![image](https://github.com/user-attachments/assets/f07e16b4-01d5-46fe-ab2d-59bd6bd029ec)
![image](https://github.com/user-attachments/assets/6d6de0b1-7f0a-4585-8c5e-3b60f5a3a33e)
![image](https://github.com/user-attachments/assets/0096e36c-fedd-46ca-b87a-a4cac1cfb8ee)
![image](https://github.com/user-attachments/assets/b65c1fb8-d986-4e65-9d9a-fc794be97ce7)
![image](https://github.com/user-attachments/assets/7b2ea056-8c96-41d2-83e0-3b7a946ccbb9)
![image](https://github.com/user-attachments/assets/f94a4308-2913-4db7-ab4e-3bf806ab9f78)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


