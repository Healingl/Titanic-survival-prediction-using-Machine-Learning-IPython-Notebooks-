
# coding: utf-8

# # Ensemble Modelling to solve the Titanic problem

# ## 1. Introduction 

# In this kernel we will solve the Titanic problem using feature engineering and ensemble modelling. The script follows 3 main parts:
# 1. Feature Analysis
# 2. Feature Engineering
# 3. Modelling

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, cross_val_score, 
StratifiedKFold, learning_curve)

sns.set(style='white', context='notebook', palette='deep')


# ## 2. Load and check data

# ### 2.1 Load Data

# In[3]:


# Load data

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
IDtest=test['PassengerId']


# ### 2.2 Outlier Detection

# In[4]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# Since outliers can have a dramatic effect on the prediction, I managed them. I used the Turkey method to detect outliers.
# 

# In[5]:


train.loc[Outliers_to_drop]


# Thus we have 10 outliers. The 28, 89, 342 have a very high passenger fare. The others have very high values of SibSp

# In[6]:


# Drop Outliers
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)


# ### 2.3 Joining train and test set

# In[7]:


## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# ### 2.4 Check for null and missing values

# In[8]:


# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()


# Age and Cabin features have an important part of missing values.
# 
# Survived missing values correspond to the join testing dataset (Survived column doesn't exist in test set and has been replace by NaN values when concatenating the train and test set)

# In[9]:


# Info
train.info()
train.isnull().sum()


# In[10]:


train.head()


# In[11]:


train.dtypes


# In[12]:


# Summarize data
train.describe()


# ## 3. Feature Analysis

# ### 3.1 Numerical Analysis

# In[13]:


# Co-relation matrix between numerical values
g = sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(),
               annot=True, fmt='.2f', cmap='coolwarm')


# Only fare seems to have co-relation between survival probability

# ### SibSp

# In[14]:


#Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings seem less to survive. Single passengers have a less chance of survival than passengers having 1-2 children

# ### Parch

# In[15]:


# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# ### Age

# In[16]:


# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.
# 
# We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived.
# 
# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
# 
# It seems that very young passengers have more chance to survive.

# In[17]:


# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# ### Fare

# In[18]:


dataset["Fare"].isnull().sum()


# In[19]:


# Fill fare missing values with the median value
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


# In[20]:


# Explore Fare distribution 
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled.
# 
# In this case, it is better to transform it with the log function to reduce this skew.

# In[21]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[22]:


g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# Skewness is clearly reduced after the log transformation

# ### 3.2 Categorical Values

# ### Sex 

# In[23]:


g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# In[24]:


train[["Sex","Survived"]].groupby('Sex').mean()


# It is clearly obvious that Male have less chance to survive than Female.
# 
# So Sex, might play an important role in the prediction of the survival.
# 

# ### PClass

# In[25]:


# Explore Pclass vs Survived
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[26]:


# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.
# 
# This trend is conserved when we look at both male and female passengers.

# ### Embarked

# In[27]:


dataset["Embarked"].isnull().sum()


# In[28]:


#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[29]:


# Explore Embarked vs Survived 
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.
# 
# My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
# 
# Let's see the Pclass distribution vs Embarked

# In[30]:


# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# ndeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.
# 
# At this point, i can't explain why first class has an higher survival rate. My hypothesis is that first class passengers were prioritised during the evacuation due to their influence.

# ## 4. Missing Values

# ### 4.1 Age

# As we see, ther are 256 missing values in the age dataset. It is preferable to keep the age feature and impute the missing values

# In[31]:


dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[32]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
# 
# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
# 
# So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
# 
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# In[33]:


# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med


# ## 5. Feature Engineering

# ### 5.1 Name

# In[34]:


dataset["Name"].head()


# The Name feature contains information on passenger's title.
# 
# Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.

# In[35]:


# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[36]:


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

# In[37]:


# Convert to categorail values title

dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Capt',
                   'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',
                    'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].map({'Master':0,'Miss':1,'Ms':1,'Mme':1,
                    'Mlle':1,'Mrs':1,'Mr':2,'Rare':3})

dataset['Title'] = dataset['Title'].astype(int)


# In[38]:


g = sns.countplot(dataset['Title'])
g = g.set_xticklabels(['Master', 'Miss/mlle,mrs','Mr','Rare'])


# In[39]:


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# It is interesting to note that passengers with rare title have more chance to survive.

# In[40]:


# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# ### 5.2 Family Size

# We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, i choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger).

# In[41]:


# Create a family size descriptor from SibSp and Parch
dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[42]:


g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# The family size seems to play an important role, survival probability is worst for large families.
# 
# Additionally, i decided to created 4 categories of family size.

# In[43]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[44]:


g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# This shows us that small and medium families have a higher chance of survival than single and large families

# In[45]:


# convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")


# In[46]:


dataset.head()


# ### 5.3 Cabin

# In[47]:


dataset['Cabin'].head()


# In[48]:


dataset['Cabin'].describe()


# In[49]:


dataset["Cabin"].isnull().sum()


# The Cabin feature column contains 292 values and 1007 missing values.
# 
# I supposed that passengers without a cabin have a missing value displayed instead of the cabin number.

# In[50]:


dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[51]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# The first letter of the cabin indicates the Desk, i choosed to keep this information only, since it indicates the probable location of the passenger in the Titanic.

# In[52]:


g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[53]:


g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# Because of the low number of passenger that have a cabin, survival probabilities have an important standard deviation and we can't distinguish between survival probability of passengers in the different desks.
# 
# But we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
# 
# It is particularly true for cabin B, C, D, E and F.

# In[54]:


dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# ### 5.4 Ticket

# In[55]:


dataset["Ticket"].head()


# In[56]:


dataset["Ticket"].describe()


# It could mean that tickets sharing the same prefixes could be booked for cabins placed together. It could therefore lead to the actual placement of the cabins within the ship.
# 
# Tickets with same prefixes may have a similar class and survival.
# 
# So i decided to replace the Ticket feature column by the ticket prefixe. Which may be more informative

# In[57]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[58]:


dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# In[59]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")


# In[60]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[61]:


dataset.head()


# ## 6. Modelling 

# In[62]:


## Separate train dataset and test dataset

train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[63]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)


# ### 6.1 Simple Modelling

# #### 6.1.1 Cross Validate Models

# In[64]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[65]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# I decided to choose the SVC, AdaBoost, RandomForest , ExtraTrees and the GradientBoosting classifiers for the ensemble modeling.

# #### 6.1.2 Hyperparameter Tuning for best models

# I performed a grid search optimization for AdaBoost, ExtraTrees, RandomForest, GradientBoosting and SVC classifiers.

# In[67]:


# AdaBoost

DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_


# In[68]:


gsadaDTC.best_score_


# In[69]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[70]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[71]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[72]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# #### 6.1.3 Plot Learning Curves

# Learning curves are a good way to see the overfitting effect on the training set and the effect of training size on accuracy

# In[74]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# Gradient Boosting and AdaBoost Classifiers tend to overfit the training set. According to the growing cross-validation curves GradientBoosting and AdaBoost could perform well with more training examples. Svc and ExtraTrees classifiers seem to better generalize the predicition since the training and cross validation curves are close together.

# #### 6.1.4 Feature imporance of tree-based classifiers

# In[75]:


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1


# We note that the 4 classifiers have 4 different top features according to the relative importance. It means thath their predictions are not based on the same feratures. But they share common features for their classification
# 
# We can say that:
# 1. Pc_1, Pc_2, Pc_3 and fare refer to the general socail standing of the passengers
# 2. Sex and Title_2 and Title_3 refer to the gender
# 3. Age and Title_1 refer to the age
# 4. Fsize, LargeF, MedF, Single refer to the sizr of the passenger family.
# 
# According to the feature importance of the 4 classifiers, the prediction of the survival seems to be more associated with the Age, Sex, the family size and the social standing of the passengers more than the location in the boat.

# In[77]:


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# The prediction seems to be quite similar for the 5 classifiers except when adaboost is compared to the other classifiers.

# ### 6.2 Ensemble Modelling

# #### 6.2.1 Combining Models

# I choosed a voting classifier to combine the predictions coming from the 5 classifiers.
# 
# I preferred to pass the argument "soft" to the voting parameter to take into account the probability of each vote.

# In[78]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# ### 6.3 Prediction

# In[79]:


test_Survived = pd.Series(votingC.predict(test), name='Survived')
results=pd.concat([IDtest, test_Survived], axis=1)
results.to_csv('ensemble_methods.csv', index=False)


# After submitting the score we got an accuracy of 0.78 on the kaggle leaderboard
