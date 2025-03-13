'''

Breast Cancer Prediction


Daily Challenge: Breast Cancer Prediction


What you will learn
Data visualization with scatter plots.
Basic understanding of classification models
Interpreting the results.


What you will create
In this notebook, you will be using 4 classification models- Logistic Regression, K Nearest Neighbours, Random Forests and Support Vector Machines (SVM).
dataset available here

Your Task
Exploratory Data Analysis
Use pandas to load the dataset and examine the first few rows.
Check and handle the missing values.
Drop any unnecessary column
Create a Countplot to display diagnosis from magma

Data Preprocessing, Building Models and Evaluation :
counts of unique rows in the ‘diagnosis’ column
map categorical values to numerical values
Splitting the data into train and test
Implement logistic regression and print the accuracy.
Implement K Nearest Neighbours and print the accuracy.
Implement Random Forests and print the accuracy.
Implement Support Vector Machines (SVM) and print the accuracy.
Which is the best model ?


'''


import pandas as pd

df = pd.read_csv('Breast Cancer Wisconsin.csv')
print(df.head())
# df.columns

# we can drop the ID column - we don't need it
df.drop(['id'], axis=1, inplace=True)
# and we can drop the 'Unnamed: 32' column because it's completely empty.
df.drop(['Unnamed: 32'], axis=1, inplace=True)

# there are no missing values in this dataset
df.isnull().sum()

# ~~~~~~~~

# The 'diagnosis' column is the target.
target = 'diagnosis'

# let's visualize the 'diagnosis' column
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df[target], palette='magma')
# sns.countplot(x='diagnosis', data=df)  # or like this
plt.xlabel("diagnosis")
plt.ylabel('Cases count')
plt.show()

# ~~~~~~~~

# the 'diagnosis' column is the target:
df[target].value_counts()

'''
M = Malignant
B = Benign
'''

# let's map these to 1 and 0
df[target] = df[target].map({'M':1, 'B':0})


# ~~~~~~~~

# split the data into X and y
X = df.drop([target], axis=1)  # all columns except 'diagnosis'
y = df[target]  # the 'diagnosis' column

# split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ~~~~~~~~

# Implement logistic regression and print the accuracy.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Accuracy (Logistic Regression): {accuracy_logreg:.2f}')

# ~~~~~~~~

# Implement K Nearest Neighbours and print the accuracy.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # training
y_pred_knn = knn.predict(X_test)  # testing

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy (KNN): {accuracy_knn:.2f}')

# ~~~~~~~~

# Implement Random Forests and print the accuracy.
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy (Random Forest): {accuracy_rf:.2f}')

# ~~~~~~~~

# Implement Support Vector Machines (SVM) and print the accuracy.
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy (SVM): {accuracy_svm:.2f}')

# ~~~~~~~~

# So Logistic Regression and Random Forest got the best accuracy: 0.95 .


