# -*- coding: utf-8 -*-
"""Mobile_prices_predction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G5nd8CWc8c94lAu2frU6pAKloISOiW_z
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing datasets
train_df = pd.read_csv('/content/mobile_price_training_data.csv')
test_df = pd.read_csv('/content/mobile_price_test_data.csv')

#checking the datasets heads
train_df.head()

test_df.head()

#seeing the info looking for nulls
train_df.info()

#test is good no nulls
test_df.info()

#this is the dataset descrition
train_df.describe()

test_df.describe()

#the nulls in the train data sum
train_df.isnull().sum()

#checking for duplicated data
train_df.duplicated().any()

"""#preprocessing

"""

#remove the nulls from the training dataset
train_df.dropna(inplace=True)

train_df.isnull().sum()

"""#Data visualization and analysis

#ram effect on the price
"""

sns.jointplot(x='ram',y='price_range',data=train_df ,color='red',kind='kde');

"""#internal memory effect on the price"""

sns.pointplot(x='price_range',y='int_memory',data=train_df,color='blue')

"""#Phones That Support 3G

"""

lables = ["3G-supported","Not supported"]
count_values = train_df['three_g'].value_counts().values

figuer1, ax1 = plt.subplots()
ax1.pie(count_values, labels=lables, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.show()

"""#Phones That Support 4G"""

lables_4G = ["4G-supported","Not supported"]
count_values_4G = train_df['four_g'].value_counts().values

fig2, ax2 = plt.subplots()
ax2.pie(count_values_4G, labels=lables_4G, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.show()

"""#Battery power effect on price"""

sns.boxplot(x='price_range',y='battery_power',data=train_df)

"""#number of phones vs fornt and back camera megapixels"""

plt.figure(figsize=(10,6))
train_df['fc'].hist(alpha=0.5,color='blue',
                    bins=30,label='Front Camera')
train_df['pc'].hist(alpha=0.5,color='red',
                    bins=30,label='Back Camera')

"""#talk time effect on price"""

sns.boxplot(x='price_range',y='talk_time',data=train_df)

"""#Weight effect on price"""

sns.boxenplot(x='price_range',y='mobile_wt',data=train_df)

"""x and y dataset vectors"""

X = train_df.drop('price_range',axis=1)
y = train_df['price_range']

"""#NOW splitting the data"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

"""#**Building models**

#Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
logic_Mo = LogisticRegression()

"""now using fit to train the model on this dataset so it can make predction"""

logic_Mo.fit(X_train,y_train)

logic_Mo.score(X_test,y_test)

"""it's a bad score trying another method

#**Decision Tree model**
"""

from sklearn.tree import DecisionTreeClassifier
dt_Mo = DecisionTreeClassifier()

dt_Mo.fit(X_train,y_train)

dt_Mo.score(X_test,y_test)

"""this result is much better but i will try more models then choose the best score.

#**Random Tree model**
"""

from sklearn.ensemble import RandomForestClassifier
rf_Mo = RandomForestClassifier(n_estimators=500)

"""n_estimators means that the algorithm will build 400 decision tress"""

rf_Mo.fit(X_train,y_train)

rf_Mo.score(X_test,y_test)

"""1. 500 gave 0.884
2. 400 gave 0.879
3. 300 gave 0.876
4. 200 gave 0.871

#**Linear Regression**
"""

from sklearn.neighbors import KNeighborsClassifier
kn_Mo = KNeighborsClassifier(n_neighbors=10)

kn_Mo.fit(X_train,y_train)

kn_Mo.score(X_test,y_test)

"""##KNN is the best model to use"""

y_prediction = kn_Mo.predict(X_test)

plt.scatter(y_test,y_prediction)

from sklearn.metrics import classification_report, confusion_matrix

predction = (kn_Mo.predict(X_test))

classification_report(y_test,predction)

matrix = confusion_matrix(y_test,predction)

matrix

plt.figure(figsize=(10,6))
sns.heatmap(matrix,annot=True)

"""#now try to predict the price using test data"""

test_df.head()

"""drop id"""

# ican put inplace=True and it will drop from the data file too
test_df = test_df.drop('id',axis=1)

test_df.head()

predicted_phone_price = kn_Mo.predict(test_df)

predicted_phone_price

"""#**END**"""

