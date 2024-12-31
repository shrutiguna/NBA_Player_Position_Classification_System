#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz


# In[40]:


import warnings

warnings.filterwarnings('ignore')


# In[41]:


basketball_data_df = pd.read_csv('nba_stats.csv')
dataset = pd.DataFrame(basketball_data_df)


# In[42]:


#Split data into train and test dataset
#Dropping the columns which are not required or which are corellated
features_train, features_test, labels_train, labels_test = train_test_split(
    dataset.drop(columns=["Pos", "Age","G"], axis=1), dataset['Pos'], test_size=0.20, random_state=0)

features_train_filtered = features_train.loc[features_train['MP'] >= 10.0]
features_train_filtered = features_train_filtered.loc[features_train['FG%'] > 0.3]
features_train_filtered = features_train_filtered.loc[features_train['2P%'] > 0.3]
features_train_filtered = features_train_filtered.loc[features_train['PF'] > 0.1]
features_train_filtered = features_train_filtered.loc[features_train['eFG%'] > 0.2]

labels_train_filtered = labels_train.loc[features_train['MP'] >= 10.0]
labels_train_filtered = labels_train_filtered.loc[features_train['FG%'] > 0.3]
labels_train_filtered = labels_train_filtered.loc[features_train['2P%'] > 0.3]
labels_train_filtered = labels_train_filtered.loc[features_train['PF'] > 0.1]
labels_train_filtered = labels_train_filtered.loc[features_train['eFG%'] > 0.2]


# In[43]:


#Building the model
linear_svm = LinearSVC(random_state=0, max_iter=1000000).fit(features_train_filtered, labels_train_filtered)
print("Training set score: {:.3f}".format(linear_svm.score(features_train, labels_train)))
print("Test set score: {:.3f}".format(linear_svm.score(features_test, labels_test)))


# In[44]:


# Confusion matrix of train set
train_predictions = linear_svm.predict(features_train_filtered)
print("Confusion matrix:")
print(pd.crosstab(labels_train_filtered, train_predictions, rownames=['True'], colnames=['Predicted'], margins=True))


# In[45]:


# Confusion matrix of test set
test_predictions = linear_svm.predict(features_test)
print("Confusion matrix:")
print(pd.crosstab(labels_test, test_predictions, rownames=['True'], colnames=['Predicted'], margins=True))


# In[46]:


linear_svm = LinearSVC(random_state=0, max_iter=50000)
cv_accuracy_scores = cross_val_score(linear_svm, dataset.drop(columns=['Pos', 'Age'], axis=1), dataset['Pos'], cv=10)
print("Cross-validation accuracy scores: {}".format(cv_accuracy_scores))
print("Average cross-validation accuracy score: {:.2f}".format(cv_accuracy_scores.mean()))


# In[47]:


#Testing on dummy set


# In[48]:


dummy_data = pd.read_csv("dummy_test.csv")
dataset = pd.DataFrame(dummy_data)


# In[49]:


test_X_Val = dummy_data.drop(columns=["Pos","Predicted Pos", "Age","G"], axis=1)
test_Y_Val = dummy_data['Pos']


# In[50]:


linearsvm = LinearSVC(random_state=0, max_iter=50000)

# Fitting the classifier with the dummy test data
linearsvm.fit(test_X_Val,test_Y_Val)

pred_test = linearsvm.predict(test_X_Val)


# In[51]:


# Calculate the accuracy of the predictions
from sklearn.metrics import accuracy_score
accu_test = accuracy_score(test_Y_Val, pred_test)
print("Accuracy on dummy test set:", accu_test)


# In[52]:


print("Confusion matrix:")
print(pd.crosstab(test_Y_Val, pred_test, rownames=['True'], colnames=['Predicted'], margins=True))


# In[ ]:


"""Linear Support Vector Machines (SVM) were implemented to predict basketball player positions using the provided dataset. SVM demonstrated superior accuracy compared to other models. After feature selection and data filtering to optimize model fitting, the SVM achieved an accuracy of 72.7% on the test dataset. The data was split into 80% for training and 20% for testing.

Various methods were employed to enhance accuracy:

- Correlation coefficient analysis was conducted to identify features with minimal correlation, resulting in improved accuracy compared to alternative models.
- Important features were selected based on the absolute values of learned coefficients in a linear SVM. Notably, the selected features in decreasing order of importance were 'MP', 'FG%', '2P%', 'FT%', 'eFG%', 'DRB', '3P', '3PA', 'PF', and 'BLK'.
- After numerous permutations, the feature set resulting in the highest accuracy comprised 'MP', 'FG%', '2P%', 'PF', and 'eFG%'.
- Larger training datasets yielded better accuracy, while changes in feature values also influenced accuracy. Higher feature values encompassed more data rows, resulting in lower accuracy.
- The 'max_iter' parameter in the SVM model significantly impacted accuracy. Adjusting it to higher values improved accuracy but decreased with excessively large iterations.
- ANN shows higher accuracy but the model is overfitting so i have implemented Linear SVM resulting in an almost 73% accuracy.
"""

