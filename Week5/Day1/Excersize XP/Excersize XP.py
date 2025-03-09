'''


Exercises XP

'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

Exercise 1 : Defining the Problem and Data Collection for Loan Default Prediction

Write a clear problem statement for predicting loan defaults.
Identify and list the types of data you would need for this project (e.g., personal details of applicants, credit scores, loan amounts, repayment history).
Discuss the sources where you can collect this data (e.g., financial institution’s internal records, credit bureaus).
Expected Output: A document detailing the problem statement and a comprehensive plan for data collection, including data types and sources.

'''

'''

Loan default is a major issue for banks.
It is in their interest to predict if a person might encounter a loan default.

They would like to have a predictive model that will assess if a person might have a loan default.
This is based on his financial data.
By analyzing the patterns, a model can assist the bank in making informed decisions and reducing losses.

Data Types Needed:
In order to build an effective model for loan default prediction, the bank will need several data types:
Personal Details: age, gender, marital status.
Employment details: job type, how long does he work.
At the bank: monthly income, amount of expenses, bank account debt, credit card score, history of loans, history of savings at the bank.
About the loan: Loan amount, interest rate, loan term (how many years), loan purpose.


Data Sources:

1.
Bank records:
Bank account records, debt history, loans history, repayment history.
History of approved or rejected loans.

2.
Credit card records:
Credit card companies that hold information about the customer.

3.
Government finantial data:
Interest rate, and yearly inflation.


'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

Exercise 2 : Feature Selection and Model Choice for Loan Default Prediction

From this dataset, identify which features might be most relevant for predicting loan defaults.
Justify your choice of features.

'''

'''

Not much important:
* Gender - not much important.
* Education - not much important - because we know the "ApplicantIncome"
* Property_Area - not much important - because we know the "LoanAmount"

* Loan_Status - this is just the column that says if the loan was approved or not - this is the target column

These are the features that are most relevant:
* Married
* Dependents
* Self_Employed
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History

These features influence the financial status of the person, and the chance of returning or failing to return the loan.

'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 3 : Training, Evaluating, and Optimizing the Model

Which model(s) would you pick for a Loan Prediction ?
Outline the steps to evaluate the model’s performance, mentioning specific metrics that would be relevant to evaluate the model.

'''

'''

Models for Loan Prediction (a loan default):
----------------------------------------------

1. Logistic regression
The first step is to start with Logistic Regression.
Because it's a simple and interpretable model that can see a linear relationship between features.
It can give us a yes/no answer (loan default or not).

2. Decision trees
A non-linear model that builds a tree of decisions.
It splits the data based on feature thresholds.

3. Gradient Boosting Models
They sequentially build decision trees to reduce prediction errors.



Steps to evaluate the model performance:
------------------------------------------

1. Prepare the data.
2. Split it into training, validation, and test sets.
3. Train the model using the training set.

4. Evaluate the loan default issue (it's a classification problem) using several metrics:
(positive prediction means that there will be a loan default)

A. Accuracy (overall percentage of correctly classified samples).  This is only if the dataset is balanced.

B. Precision (proportion of correct predictions out of all default loan predictions).
(We might refuse to give loans to people who could have been good customers)

C. Recall (Sensitivity):
Measures how much we were right about the actual defaults.
(We don't want to have many False-Negatives - people mistakely classified as non-defaulters - because this is a high risk for the bank).

D. F1 Score: an harmonic mean of precision and recall.  This is a balance between the two scores.

E. Area Under the ROC Curve (AUC-ROC):
Measures the model's ability to distinguish between yes and no (defaulters and non-defaulters) - by going through all thresholds.
High AUC means better ability to distinguish between yes and no.

F. Confusion matrix: It gives you a matrix of TP, FP, FN, TN.  This will enable you to examine the classification performance.


'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 4 : Designing Machine Learning Solutions for Specific Problems

For each of these scenario, decide which type of machine learning would be most suitable. Explain.

Predicting Stock Prices : predict future prices
Organizing a Library of Books : group books into genres or categories based on similarities.
Program a robot to navigate and find the shortest path in a maze.


'''
'''

1.
For predicting stock prices : predict future prices.

Linear regression and decision trees will not be that good for predicting stock prices.
This is because predicting stock prices is very complex - it's highly dynamic and it's not linear.

So a better approach will be "Reinforcement learning".
The model will learn to take actions on the market, based on the market situation.

You can also use clustering algorithms to group similar stocks on the market.
Or you can find patterns in the data.
Such an algorithm is K-Means.



2.
Organizing a Library of Books : group books into genres or categories based on similarities.

For this we would prefer unsupervised learning, because it's about finding patterns, structures, and clusters in unlabeled data.
The model will have to find the genres, because the books are unlabled (you don't know what the genres are).
The machine needs to identify the similarities between the books.

So we would need clustering algorithms (like K-Means clustering).


3.
Program a robot to navigate and find the shortest path in a maze.

The best approach would be "Reinforcement Learning".

The robot will have to find the shortest path.
We will give him rewards on taking the correct turn.




'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 5 : Designing an Evaluation Strategy for Different ML Models

Select three types of machine learning models: one from supervised learning (e.g., a classification model), one from unsupervised learning (e.g., a clustering model), and one from reinforcement learning.

For the supervised model, outline a strategy to evaluate its performance, including the choice of metrics (like accuracy, precision, recall, F1-score) and methods (like cross-validation, ROC curves).

For the unsupervised model, describe how you would assess the effectiveness of the model, considering techniques like silhouette score, elbow method, or cluster validation metrics.

For the reinforcement learning model, discuss how you would measure its success, considering aspects like cumulative reward, convergence, and exploration vs. exploitation balance.

Address the challenges and limitations of evaluating models in each category.


'''


'''

We didn't study any of the effectiveness assessments mentioned above, and any of the other algorithms.
So I will do only the first part, with the supervised learning.


Let's choose "Logistic Regression", as a classification model for supervised learning.

Evaluation metrics:
After training the model, we need to evaluate its performance.

A. Accuracy:
Measures the proportion of correctly classified samples: TP+TN divided by all.

B. Precision:
The proportion of true positives (TP) out of all samples predicted as positive: TP/(TP+FP)
Useful when FP are costly (like in fraud detection).

C. Recall (Sensitivity):
The proportion of TP out of all actual positive samples: TP/(TP+FN)
Useful when FN are costly (like in a medical desease diagnosis).

D. F1-Score:
The harmonic mean of precision and recall.
Useful for imbalanced datasets where both precision and recall need to be balanced.

E. ROC-AUC:
Measures the model's ability to distinguish between positive and negative classes across different thresholds.
AUC (Area Under the Curve) provides a single score that summarizes the classifier's performance:
AUC = 1: perfect classifier.
AUC = 0.5: not better than just random guessing.

F. Confusion matrix:
Provides a detailed view (a matrix) of TP,TN,FP,FN.
This is good for identifing errors in predictions.



Cross-Validation (an evaluation method).

Cross-validation is a method used to assess how well a model can work with unseen data.
Instead of using only a single train-test split, cross-validation ensures that the model is evaluated on multiple subsets of the data.
So making the evaluation more reliable.

For example, the k-Fold Cross-Validation.

1. The data is split into k subsets (folds).

2. Training and Testing: The model is trained on a portion of the data and validated on the remaining portion.
This process is repeated multiple times with different subsets.

3. Doing an average of the performance metrics (like accuracy, precision, recall, etc.) on each fold.



'''

